import { PerformanceManager } from './performance';

export interface FaceDetectorEvent {
  faceCount: number;
  yawDeg: number;
  pitchDeg: number;
  frameDiff: number;
  keypoints: Array<{ x: number; y: number }> | null;
  latencyMs: number;
  timestamp: number;
}

export interface PipelineCallbacks {
  onFaceResult?: (data: FaceDetectorEvent) => void;
  onWorkerReady?: (delegate: string) => void;
  onError?: (error: string) => void;
}

export class CameraPipeline {
  private video: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private faceWorker: Worker | null = null;

  private isRunning = false;
  private faceWorkerInFlight = false;
  private timerId: ReturnType<typeof setTimeout> | null = null;

  readonly performance = new PerformanceManager();
  private callbacks: PipelineCallbacks = {};

  async init(videoElement: HTMLVideoElement, callbacks: PipelineCallbacks = {}) {
    this.video = videoElement;
    this.callbacks = callbacks;

    // Low-power webcam stream: 320x240 @ 10-15 fps
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 320 },
        height: { ideal: 240 },
        frameRate: { ideal: 10, max: 15 },
        facingMode: 'user',
      },
      audio: false,
    });

    this.video.srcObject = this.stream;
    await this.video.play();

    this.faceWorker = new Worker(
      new URL('../workers/face-detector.worker.ts', import.meta.url),
      { type: 'module' }
    );

    this.faceWorker.onmessage = (e) => {
      const { data } = e;
      if (data.type === 'INIT_SUCCESS') {
        this.callbacks.onWorkerReady?.(data.delegate);
        this.scheduleNextTick(0);
      } else if (data.type === 'INIT_ERROR') {
        this.callbacks.onError?.(data.error);
      } else if (data.type === 'FACE_RESULT') {
        this.faceWorkerInFlight = false;
        if (data.success) {
          this.performance.recordFaceLatency(data.latencyMs);
          this.callbacks.onFaceResult?.(data);
        } else {
          this.callbacks.onError?.(data.error);
        }
        if (this.isRunning) {
          this.scheduleNextTick(this.performance.getFaceIntervalMs());
        }
      }
    };

    this.faceWorker.postMessage({
      type: 'INIT',
      wasmPath: '/wasm',
      modelPath: '/models/blaze_face_short_range.tflite',
    });

    this.isRunning = true;
  }

  private scheduleNextTick(delayMs: number) {
    if (!this.isRunning) return;
    if (this.timerId !== null) clearTimeout(this.timerId);
    this.timerId = setTimeout(this.dispatchFrame, delayMs);
  }

  private dispatchFrame = () => {
    if (!this.isRunning || !this.video || this.faceWorkerInFlight) return;

    if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      this.faceWorkerInFlight = true;
      const now = performance.now();
      createImageBitmap(this.video)
        .then((bitmap) => {
          this.faceWorker?.postMessage(
            { type: 'PROCESS_FRAME', bitmap, timestamp: now },
            [bitmap]
          );
        })
        .catch((err) => {
          this.faceWorkerInFlight = false;
          console.error('Failed to create ImageBitmap:', err);
          this.scheduleNextTick(this.performance.getFaceIntervalMs());
        });
    } else {
      this.scheduleNextTick(250);
    }
  };

  stop() {
    this.isRunning = false;
    if (this.timerId !== null) {
      clearTimeout(this.timerId);
      this.timerId = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    if (this.faceWorker) {
      this.faceWorker.terminate();
      this.faceWorker = null;
    }
  }
}
