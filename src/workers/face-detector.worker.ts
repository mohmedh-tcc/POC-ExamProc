import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

let detector: FaceDetector | null = null;
let isInitializing = false;
let lastTimestamp = 0;

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

async function initFaceDetector(wasmPath: string, modelPath: string) {
  if (detector || isInitializing) return;
  isInitializing = true;

  try {
    const vision = await FilesetResolver.forVisionTasks(wasmPath, true);
    detector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath,
        delegate: 'CPU', // CPU WASM avoids waking GPU at 1-2 Hz
      },
      runningMode: 'VIDEO',
      minDetectionConfidence: 0.5,
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
    await initFaceDetector(data.wasmPath, data.modelPath);
    return;
  }

  if (data.type === 'PROCESS_FRAME') {
    const bitmap: ImageBitmap = data.bitmap;
    const clientTimestamp: number = data.timestamp;
    const startTime = performance.now();

    if (!detector) {
      bitmap.close();
      self.postMessage({
        type: 'FACE_RESULT',
        success: false,
        error: 'FaceDetector not initialized',
      });
      return;
    }

    try {
      const frameDiff = computeFrozenDiff(bitmap);
      lastTimestamp = Math.max(lastTimestamp + 1, Math.round(clientTimestamp));
      const results = detector.detectForVideo(bitmap, lastTimestamp);
      const latencyMs = performance.now() - startTime;

      const detections = results.detections ?? [];
      const faceCount = detections.length;

      let yawDeg = 0;
      let pitchDeg = 0;
      let keypoints: Array<{ x: number; y: number }> | null = null;

      if (faceCount > 0 && detections[0].keypoints && detections[0].keypoints.length >= 6) {
        const kps = detections[0].keypoints;
        // 0: right eye, 1: left eye, 2: nose tip, 3: mouth center, 4: right ear, 5: left ear
        const rightEye = kps[0];
        const leftEye = kps[1];
        const nose = kps[2];
        const mouth = kps[3];
        const rightEar = kps[4];
        const leftEar = kps[5];

        const earSpan = Math.max(0.02, Math.abs(leftEar.x - rightEar.x));
        const yawRatio = ((nose.x - rightEar.x) - (leftEar.x - nose.x)) / earSpan;
        yawDeg = yawRatio * 45;

        const eyeMidY = (rightEye.y + leftEye.y) / 2;
        const faceH = Math.max(0.02, mouth.y - eyeMidY);
        const pitchRatio = (nose.y - eyeMidY) / faceH;
        // Neutral pitchRatio is ~0.58; higher means nose dropped toward mouth (looking down)
        pitchDeg = (0.58 - pitchRatio) * 60;

        keypoints = kps.map((k) => ({ x: k.x, y: k.y }));
      }

      self.postMessage({
        type: 'FACE_RESULT',
        success: true,
        faceCount,
        yawDeg,
        pitchDeg,
        frameDiff,
        keypoints,
        latencyMs,
        timestamp: clientTimestamp,
      });
    } catch (err) {
      self.postMessage({
        type: 'FACE_RESULT',
        success: false,
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      bitmap.close();
    }
  }
};
