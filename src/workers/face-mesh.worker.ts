import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { decomposeTransformMatrix, EulerAngles } from '../lib/math/transform-matrix';
import { computeIrisGaze, IrisGazeResult, NormalizedLandmark } from '../lib/math/iris-gaze';
import { OneEuroFilter } from '../lib/math/one-euro';

let landmarker: FaceLandmarker | null = null;
let isInitializing = false;
let lastTimestamp = 0;

// One-Euro filters for stabilizing output angles and iris ratios against jitter
const pitchFilter = new OneEuroFilter(1.0, 0.007);
const yawFilter = new OneEuroFilter(1.0, 0.007);
const rollFilter = new OneEuroFilter(1.0, 0.007);
const hRatioFilter = new OneEuroFilter(1.0, 0.007);
const vRatioFilter = new OneEuroFilter(1.0, 0.007);
const dxFilter = new OneEuroFilter(1.0, 0.007);
const dyFilter = new OneEuroFilter(1.0, 0.007);

let previousGaze: IrisGazeResult | undefined = undefined;

// 8x8 grayscale thumbnail for frozen-frame detection
let prevThumb: Uint8Array | null = null;
let thumbCanvas: OffscreenCanvas | null = null;
let thumbCtx: OffscreenCanvasRenderingContext2D | null = null;

function computeFrozenDiff(bitmap: ImageBitmap): number {
  if (!thumbCanvas) {
    thumbCanvas = new OffscreenCanvas(8, 8);
    thumbCtx = thumbCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (!thumbCtx) return 255;

  thumbCtx.drawImage(bitmap, 0, 0, 8, 8);
  const data = thumbCtx.getImageData(0, 0, 8, 8).data;
  const curr = new Uint8Array(64);
  for (let i = 0; i < 64; i++) {
    const p = i * 4;
    curr[i] = Math.round(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]);
  }

  if (!prevThumb) {
    prevThumb = curr;
    return 255;
  }

  let sum = 0;
  for (let i = 0; i < 64; i++) {
    sum += Math.abs(curr[i] - prevThumb[i]);
  }
  prevThumb = curr;
  return sum / 64;
}

// Rolling Mouth Aspect Ratio (MAR) buffer for visual speaking detection (~2s window @ 5 Hz)
const marHistory: number[] = [];
const MAR_WINDOW = 10;

function computeMouthMotion(landmarks: NormalizedLandmark[]): {
  mar: number;
  isSpeaking: boolean;
  motionScore: number;
} {
  const upperLip = landmarks[13];
  const lowerLip = landmarks[14];
  const leftCorner = landmarks[61];
  const rightCorner = landmarks[291];

  const mouthWidth = Math.max(
    0.01,
    Math.hypot(rightCorner.x - leftCorner.x, rightCorner.y - leftCorner.y)
  );
  const lipGap = Math.hypot(lowerLip.x - upperLip.x, lowerLip.y - upperLip.y);
  const mar = lipGap / mouthWidth;

  marHistory.push(mar);
  if (marHistory.length > MAR_WINDOW) {
    marHistory.shift();
  }

  // Count real open/close cycles using a 0.045 MAR hysteresis to ignore landmark jitter
  // and static open-mouth postures.
  const HYST = 0.045;
  let transitions = 0;
  let extremum = marHistory[0];
  let maxMar = marHistory[0];
  let direction: 0 | 1 | -1 = 0; // 1 = opening, -1 = closing

  for (let i = 1; i < marHistory.length; i++) {
    const v = marHistory[i];
    if (v > maxMar) maxMar = v;

    if (direction === 0) {
      if (v - extremum >= HYST) {
        direction = 1;
        extremum = v;
        transitions++;
      } else if (extremum - v >= HYST) {
        direction = -1;
        extremum = v;
        transitions++;
      }
    } else if (direction === 1) {
      if (v > extremum) {
        extremum = v;
      } else if (extremum - v >= HYST) {
        direction = -1;
        extremum = v;
        transitions++;
      }
    } else {
      if (v < extremum) {
        extremum = v;
      } else if (v - extremum >= HYST) {
        direction = 1;
        extremum = v;
        transitions++;
      }
    }
  }

  // Speaking requires >= 3 open/close transitions in the ~2.5s window and meaningful opening
  const isSpeaking = marHistory.length >= 6 && transitions >= 3 && maxMar > 0.06;

  return { mar, isSpeaking, motionScore: transitions };
}

async function initFaceLandmarker(wasmPath: string, modelPath: string) {
  if (landmarker || isInitializing) return;
  isInitializing = true;

  try {
    const vision = await FilesetResolver.forVisionTasks(wasmPath, true);

    landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath,
        delegate: 'CPU', // CPU WASM avoids waking GPU at 4-5 Hz
      },
      runningMode: 'VIDEO',
      numFaces: 2,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      outputFacialTransformationMatrixes: true,
      outputFaceBlendshapes: false,
    });
    self.postMessage({ type: 'INIT_SUCCESS', delegate: 'CPU' });
  } catch (err) {
    self.postMessage({
      type: 'INIT_ERROR',
      error: err instanceof Error ? err.message : String(err),
    });
  } finally {
    isInitializing = false;
  }
}

self.onmessage = async (e: MessageEvent) => {
  const { data } = e;

  if (data.type === 'INIT') {
    await initFaceLandmarker(data.wasmPath, data.modelPath);
    return;
  }

  if (data.type === 'PROCESS_FRAME') {
    const bitmap: ImageBitmap = data.bitmap;
    const clientTimestamp: number = data.timestamp;
    const startTime = performance.now();

    if (!landmarker) {
      bitmap.close();
      self.postMessage({
        type: 'FACE_RESULT',
        success: false,
        error: 'FaceLandmarker not initialized',
      });
      return;
    }

    try {
      const frameDiff = computeFrozenDiff(bitmap);
      // Ensure strictly increasing integer timestamp required by MediaPipe VIDEO mode
      lastTimestamp = Math.max(lastTimestamp + 1, Math.round(clientTimestamp));
      const results = landmarker.detectForVideo(bitmap, lastTimestamp);
      const latencyMs = performance.now() - startTime;

      const faceCount = results.faceLandmarks.length;

      let angles: EulerAngles | null = null;
      let rawAngles: EulerAngles | null = null;
      let transformMatrix: number[] | undefined = undefined;
      let gaze: IrisGazeResult | null = null;
      let mouth: { mar: number; isSpeaking: boolean; motionScore: number } | null = null;
      let landmarks: NormalizedLandmark[] | null = null;

      if (faceCount > 0) {
        landmarks = results.faceLandmarks[0] as NormalizedLandmark[];

        // 1. Head Pose Euler decomposition + raw 4x4 matrix
        if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
          const matrix = results.facialTransformationMatrixes[0];
          transformMatrix = Array.from(matrix.data);
          rawAngles = decomposeTransformMatrix(matrix.data);

          // Apply One-Euro filter to damp jitter from disabled single-face smoothing
          angles = {
            pitch: pitchFilter.filter(rawAngles.pitch, lastTimestamp),
            yaw: yawFilter.filter(rawAngles.yaw, lastTimestamp),
            roll: rollFilter.filter(rawAngles.roll, lastTimestamp),
          };
        }

        // 2. Aspect-corrected, yaw-foreshortening-compensated Iris Gaze estimation
        if (landmarks) {
          const aspectRatio = bitmap.width / Math.max(1, bitmap.height);
          const headYawRad = angles ? (angles.yaw * Math.PI) / 180 : 0;
          const rawGaze = computeIrisGaze(landmarks, previousGaze, {
            aspectRatio,
            headYawRad,
          });
          if (!rawGaze.isBlinking) {
            gaze = {
              ...rawGaze,
              hRatio: hRatioFilter.filter(rawGaze.hRatio, lastTimestamp),
              vRatio: vRatioFilter.filter(rawGaze.vRatio, lastTimestamp),
              dx: dxFilter.filter(rawGaze.dx, lastTimestamp),
              dy: dyFilter.filter(rawGaze.dy, lastTimestamp),
              isBlinking: false,
            };
          } else {
            gaze = rawGaze;
          }
          previousGaze = gaze;
          mouth = computeMouthMotion(landmarks);
        }
      } else {
        marHistory.length = 0;
      }

      self.postMessage({
        type: 'FACE_RESULT',
        success: true,
        faceCount,
        angles,
        rawAngles,
        transformMatrix,
        gaze,
        mouth,
        frameDiff,
        latencyMs,
        timestamp: clientTimestamp,
        // Send key landmarks only for lightweight wireframe rendering
        keyPoints: landmarks ? [
          landmarks[33], landmarks[133], landmarks[362], landmarks[263], // Eye corners
          landmarks[468], landmarks[473], // Irises
          landmarks[1], landmarks[4], // Nose tip
          landmarks[61], landmarks[291], landmarks[13], landmarks[14], // Mouth corners + inner lips
        ] : null,
      });
    } catch (err) {
      self.postMessage({
        type: 'FACE_RESULT',
        success: false,
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      // Critical: Immediately close the transferred ImageBitmap to prevent GPU memory leak
      bitmap.close();
    }
  }
};
