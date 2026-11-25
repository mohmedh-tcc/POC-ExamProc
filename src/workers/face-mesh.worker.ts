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

let previousGaze: IrisGazeResult | undefined = undefined;

async function initFaceLandmarker(wasmPath: string, modelPath: string) {
  if (landmarker || isInitializing) return;
  isInitializing = true;

  try {
    const vision = await FilesetResolver.forVisionTasks(wasmPath, true);

    try {
      // Attempt GPU delegate first
      landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 2,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        outputFacialTransformationMatrixes: true,
        outputFaceBlendshapes: false, // Performance optimization
      });
      self.postMessage({ type: 'INIT_SUCCESS', delegate: 'GPU' });
    } catch (gpuErr) {
      console.warn('FaceLandmarker WebGL initialization failed, falling back to CPU', gpuErr);
      landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numFaces: 2,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        outputFacialTransformationMatrixes: true,
        outputFaceBlendshapes: false,
      });
      self.postMessage({ type: 'INIT_SUCCESS', delegate: 'CPU' });
    }
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
      // Ensure strictly increasing integer timestamp required by MediaPipe VIDEO mode
      lastTimestamp = Math.max(lastTimestamp + 1, Math.round(clientTimestamp));
      const results = landmarker.detectForVideo(bitmap, lastTimestamp);
      const latencyMs = performance.now() - startTime;

      const faceCount = results.faceLandmarks.length;

      let angles: EulerAngles | null = null;
      let rawAngles: EulerAngles | null = null;
      let gaze: IrisGazeResult | null = null;
      let landmarks: NormalizedLandmark[] | null = null;

      if (faceCount > 0) {
        landmarks = results.faceLandmarks[0] as NormalizedLandmark[];

        // 1. Head Pose Euler decomposition
        if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
          const matrix = results.facialTransformationMatrixes[0];
          rawAngles = decomposeTransformMatrix(matrix.data);

          // Apply One-Euro filter to damp jitter from disabled single-face smoothing
          angles = {
            pitch: pitchFilter.filter(rawAngles.pitch, lastTimestamp),
            yaw: yawFilter.filter(rawAngles.yaw, lastTimestamp),
            roll: rollFilter.filter(rawAngles.roll, lastTimestamp),
          };
        }

        // 2. Iris Gaze estimation
        if (landmarks) {
          const rawGaze = computeIrisGaze(landmarks, previousGaze);
          if (!rawGaze.isBlinking) {
            gaze = {
              hRatio: hRatioFilter.filter(rawGaze.hRatio, lastTimestamp),
              vRatio: vRatioFilter.filter(rawGaze.vRatio, lastTimestamp),
              isBlinking: false,
              rightOpenness: rawGaze.rightOpenness,
              leftOpenness: rawGaze.leftOpenness,
            };
          } else {
            gaze = rawGaze;
          }
          previousGaze = gaze;
        }
      }

      self.postMessage({
        type: 'FACE_RESULT',
        success: true,
        faceCount,
        angles,
        rawAngles,
        gaze,
        latencyMs,
        timestamp: clientTimestamp,
        // Send key landmarks only for lightweight wireframe rendering
        keyPoints: landmarks ? [
          landmarks[33], landmarks[133], landmarks[362], landmarks[263], // Eye corners
          landmarks[468], landmarks[473], // Irises
          landmarks[1], landmarks[4], // Nose tip
          landmarks[61], landmarks[291], // Mouth corners
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
