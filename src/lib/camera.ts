import { EulerAngles } from './math/transform-matrix';
import { IrisGazeResult, NormalizedLandmark } from './math/iris-gaze';
import { PerformanceManager } from './performance';

export interface FaceDetectionEvent {
  faceCount: number;
  angles: EulerAngles | null;
  rawAngles: EulerAngles | null;
  gaze: IrisGazeResult | null;
  latencyMs: number;
  timestamp: number;
  keyPoints: NormalizedLandmark[] | null;
}

export interface ObjectDetectionEvent {
  hasPhone: boolean;
  personCount: number;
  hasBook: boolean;
  detections: Array<{
    categoryName: string;
    score: number;
    boundingBox?: any;
  }>;
  latencyMs: number;
  timestamp: number;
}

export interface PipelineCallbacks {
  onFaceResult?: (data: FaceDetectionEvent) => void;
  onObjectResult?: (data: ObjectDetectionEvent) => void;
  onWorkerReady?: (workerName: 'face' | 'object', delegate: string) => void;
  onError?: (workerName: 'face' | 'object', error: string) => void;
}

export class CameraPipeline {
  private video: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private faceWorker: Worker | null = null;
  private objectWorker: Worker | null = null;

  private isRunning = false;
  private faceWorkerInFlight = false;
  private objectWorkerInFlight = false;

  private lastFaceDispatchTime = 0;
  private lastObjectDispatchTime = 0;
  private animationFrameId: number | null = null;

  readonly performance = new PerformanceManager();
  private callbacks: PipelineCallbacks = {};

  async init(videoElement: HTMLVideoElement, callbacks: PipelineCallbacks = {}) {
    this.video = videoElement;
    this.callbacks = callbacks;

    // 1. Get webcam stream
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 },
        facingMode: 'user',
      },
      audio: false,
    });

    this.video.srcObject = this.stream;
    await this.video.play();

    // 2. Spawn Web Workers
    this.faceWorker = new Worker(
      new URL('../workers/face-mesh.worker.ts', import.meta.url),
      { type: 'module' }
    );

    this.objectWorker = new Worker(
      new URL('../workers/object-detector.worker.ts', import.meta.url),
      { type: 'module' }
    );

    // 3. Worker message listeners
    this.faceWorker.onmessage = (e) => {
      const { data } = e;
      if (data.type === 'INIT_SUCCESS') {
        this.callbacks.onWorkerReady?.('face', data.delegate);
      } else if (data.type === 'INIT_ERROR') {
        this.callbacks.onError?.('face', data.error);
      } else if (data.type === 'FACE_RESULT') {
        this.faceWorkerInFlight = false;
        if (data.success) {
          this.performance.recordFaceLatency(data.latencyMs);
          this.callbacks.onFaceResult?.(data);
        } else {
          this.callbacks.onError?.('face', data.error);
        }
      }
    };

    this.objectWorker.onmessage = (e) => {
      const { data } = e;
      if (data.type === 'INIT_SUCCESS') {
        this.callbacks.onWorkerReady?.('object', data.delegate);
      } else if (data.type === 'INIT_ERROR') {
        this.callbacks.onError?.('object', data.error);
      } else if (data.type === 'OBJECT_RESULT') {
        this.objectWorkerInFlight = false;
        if (data.success) {
          this.performance.recordObjectLatency(data.latencyMs);
          this.callbacks.onObjectResult?.(data);
        } else {
          this.callbacks.onError?.('object', data.error);
        }
      }
    };

    // 4. Send explicit INIT messages with strictly local asset paths
    this.faceWorker.postMessage({
      type: 'INIT',
      wasmPath: '/wasm',
      modelPath: '/models/face_landmarker.task',
    });

    this.objectWorker.postMessage({
      type: 'INIT',
      wasmPath: '/wasm',
      modelPath: '/models/efficientdet_lite0.tflite',
    });

    // 5. Start frame pump
    this.isRunning = true;
    this.pumpFrames();
  }

  private pumpFrames = () => {
    if (!this.isRunning || !this.video) return;

    const now = performance.now();

    // Fast-loop: Face Landmarker (target 5 Hz = 200ms)
    const faceInterval = this.performance.getFaceIntervalMs();
    if (!this.faceWorkerInFlight && now - this.lastFaceDispatchTime >= faceInterval) {
      if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        this.faceWorkerInFlight = true;
        this.lastFaceDispatchTime = now;
        createImageBitmap(this.video)
          .then((bitmap) => {
            this.faceWorker?.postMessage(
              { type: 'PROCESS_FRAME', bitmap, timestamp: now },
              [bitmap]
            );
          })
          .catch((err) => {
            this.faceWorkerInFlight = false;
            console.error('Failed to create ImageBitmap for face worker:', err);
          });
      }
    }

    // Slow-loop: Object Detector (target 2.5s = 2500ms)
    const objectInterval = this.performance.getObjectIntervalMs();
    if (!this.objectWorkerInFlight && now - this.lastObjectDispatchTime >= objectInterval) {
      if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        this.objectWorkerInFlight = true;
        this.lastObjectDispatchTime = now;
        createImageBitmap(this.video)
          .then((bitmap) => {
            this.objectWorker?.postMessage(
              { type: 'PROCESS_FRAME', bitmap, timestamp: now },
              [bitmap]
            );
          })
          .catch((err) => {
            this.objectWorkerInFlight = false;
            console.error('Failed to create ImageBitmap for object worker:', err);
          });
      }
    }

    this.animationFrameId = requestAnimationFrame(this.pumpFrames);
  };

  stop() {
    this.isRunning = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    if (this.faceWorker) {
      this.faceWorker.terminate();
      this.faceWorker = null;
    }

    if (this.objectWorker) {
      this.objectWorker.terminate();
      this.objectWorker = null;
    }
  }
}
