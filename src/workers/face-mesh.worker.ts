import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { decomposeTransformMatrix, EulerAngles } from '../lib/math/transform-matrix';
import { computeIrisGaze, IrisGazeResult, NormalizedLandmark } from '../lib/math/iris-gaze';
import { MouthMotion, MouthMotionTracker } from '../lib/math/mouth-motion';
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

const mouthTracker = new MouthMotionTracker();
let previousGaze: IrisGazeResult | undefined = undefined;

// Zero-allocation 8x8 grayscale buffers for frozen-frame detection
let prevThumb = new Uint8Array(64);
let currThumb = new Uint8Array(64);
let hasPrevThumb = false;
let prevThumbTs = 0;
let lastFrozenDiff = 255;
let thumbCanvas: OffscreenCanvas | null = null;
let thumbCtx: OffscreenCanvasRenderingContext2D | null = null;

// Compare frames at a fixed spacing so 8 Hz burst frames don't look frozen
const FROZEN_MIN_GAP_MS = 450;

function resetTrackers(): void {
  pitchFilter.reset();
  yawFilter.reset();
  rollFilter.reset();
  hRatioFilter.reset();
  vRatioFilter.reset();
  dxFilter.reset();
  dyFilter.reset();
  mouthTracker.reset();
  previousGaze = undefined;
}

function selectPrimaryFaceIndex(faceLandmarks: NormalizedLandmark[][]): number {
  if (faceLandmarks.length <= 1) return 0;
  let bestIdx = 0;
  let bestSpan = -1;
  for (let i = 0; i < faceLandmarks.length; i++) {
    const lm = faceLandmarks[i];
    const leftEye = lm[33];
    const rightEye = lm[263];
    if (!leftEye || !rightEye) continue;
    const span = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
    if (span > bestSpan) {
      bestSpan = span;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function computeFrozenDiff(bitmap: ImageBitmap, ts: number): number {
  if (hasPrevThumb && ts - prevThumbTs < FROZEN_MIN_GAP_MS) return lastFrozenDiff;
  if (!thumbCanvas) {
    thumbCanvas = new OffscreenCanvas(8, 8);
    thumbCtx = thumbCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (!thumbCtx) return 255;

  thumbCtx.drawImage(bitmap, 0, 0, 8, 8);
  const data = thumbCtx.getImageData(0, 0, 8, 8).data;
  for (let i = 0; i < 64; i++) {
    const p = i * 4;
    currThumb[i] = Math.round(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]);
  }

  prevThumbTs = ts;
  if (!hasPrevThumb) {
    const tmp = prevThumb;
    prevThumb = currThumb;
    currThumb = tmp;
    hasPrevThumb = true;
    return (lastFrozenDiff = 255);
  }

  let sum = 0;
  for (let i = 0; i < 64; i++) {
    sum += Math.abs(currThumb[i] - prevThumb[i]);
  }
  const tmp = prevThumb;
  prevThumb = currThumb;
  currThumb = tmp;
  return (lastFrozenDiff = sum / 64);
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
      const frameDiff = computeFrozenDiff(bitmap, clientTimestamp);
      // Ensure strictly increasing integer timestamp required by MediaPipe VIDEO mode
      lastTimestamp = Math.max(lastTimestamp + 1, Math.round(clientTimestamp));
      const results = landmarker.detectForVideo(bitmap, lastTimestamp);
      const latencyMs = performance.now() - startTime;

      const faceCount = results.faceLandmarks.length;

      let angles: EulerAngles | null = null;
      let rawAngles: EulerAngles | null = null;
      let transformMatrix: number[] | undefined = undefined;
      let gaze: IrisGazeResult | null = null;
      let mouth: MouthMotion | null = null;
      let landmarks: NormalizedLandmark[] | null = null;

      if (faceCount > 0) {
        const primaryIdx = selectPrimaryFaceIndex(results.faceLandmarks as NormalizedLandmark[][]);
        landmarks = results.faceLandmarks[primaryIdx] as NormalizedLandmark[];

        // 1. Head Pose Euler decomposition + raw 4x4 matrix
        const matrix = results.facialTransformationMatrixes?.[primaryIdx];
        if (matrix) {
          transformMatrix = Array.from(matrix.data);
          rawAngles = decomposeTransformMatrix(matrix.data);

          // Apply One-Euro filter to damp jitter from disabled single-face smoothing
          angles = {
            pitch: pitchFilter.filter(rawAngles.pitch, lastTimestamp),
            yaw: yawFilter.filter(rawAngles.yaw, lastTimestamp),
            roll: rollFilter.filter(rawAngles.roll, lastTimestamp),
          };
        }

        // 2. Aspect-corrected, yaw-foreshortening-compensated Iris Gaze & Mouth Motion
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
          mouth = mouthTracker.update(landmarks, bitmap.width, bitmap.height, clientTimestamp, headYawRad);
        }
      } else {
        resetTrackers();
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
