import { ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision';

let detector: ObjectDetector | null = null;
let isInitializing = false;
let lastTimestamp = 0;

async function initObjectDetector(wasmPath: string, modelPath: string) {
  if (detector || isInitializing) return;
  isInitializing = true;

  try {
    const vision = await FilesetResolver.forVisionTasks(wasmPath, true);

    try {
      // Try GPU delegate first
      detector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        scoreThreshold: 0.35,
        categoryAllowlist: ['cell phone', 'person', 'book'],
      });
      self.postMessage({ type: 'INIT_SUCCESS', delegate: 'GPU' });
    } catch (gpuErr) {
      console.warn('ObjectDetector WebGL initialization failed, falling back to CPU', gpuErr);
      detector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        scoreThreshold: 0.35,
        categoryAllowlist: ['cell phone', 'person', 'book'],
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
    await initObjectDetector(data.wasmPath, data.modelPath);
    return;
  }

  if (data.type === 'PROCESS_FRAME') {
    const bitmap: ImageBitmap = data.bitmap;
    const clientTimestamp: number = data.timestamp;
    const startTime = performance.now();

    if (!detector) {
      bitmap.close();
      self.postMessage({
        type: 'OBJECT_RESULT',
        success: false,
        error: 'ObjectDetector not initialized',
      });
      return;
    }

    try {
      lastTimestamp = Math.max(lastTimestamp + 1, Math.round(clientTimestamp));
      const results = detector.detectForVideo(bitmap, lastTimestamp);
      const latencyMs = performance.now() - startTime;

      let hasPhone = false;
      let personCount = 0;
      let hasBook = false;

      const detections = results.detections.map((d) => {
        const category = d.categories[0];
        const categoryName = category?.categoryName?.toLowerCase() ?? '';
        const score = category?.score ?? 0;

        if (categoryName === 'cell phone' && score >= 0.4) {
          hasPhone = true;
        } else if (categoryName === 'person' && score >= 0.4) {
          personCount += 1;
        } else if (categoryName === 'book' && score >= 0.4) {
          hasBook = true;
        }

        return {
          categoryName,
          score,
          boundingBox: d.boundingBox,
        };
      });

      self.postMessage({
        type: 'OBJECT_RESULT',
        success: true,
        hasPhone,
        personCount,
        hasBook,
        detections,
        latencyMs,
        timestamp: clientTimestamp,
      });
    } catch (err) {
      self.postMessage({
        type: 'OBJECT_RESULT',
        success: false,
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      bitmap.close();
    }
  }
};
