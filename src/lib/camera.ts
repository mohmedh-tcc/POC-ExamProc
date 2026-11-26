import { PerformanceManager } from './performance';
import { EulerAngles } from './math/transform-matrix';
import { IrisGazeResult, NormalizedLandmark } from './math/iris-gaze';

export interface MouthMotionEvent {
  mar: number;
  isSpeaking: boolean;
  motionScore: number;
  isActive: boolean;
}

export interface FaceMeshEvent {
  faceCount: number;
  angles: EulerAngles | null;
  rawAngles: EulerAngles | null;
  transformMatrix?: number[];
  gaze: IrisGazeResult | null;
  mouth: MouthMotionEvent | null;
  frameDiff: number;
  keyPoints: NormalizedLandmark[] | null;
  latencyMs: number;
  timestamp: number;
}

export interface PipelineCallbacks {
  onFaceResult?: (data: FaceMeshEvent) => void;
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
  private lastDispatchAt = 0;
  // Warm-up and calibration collect one sample per result, so they always sample at the fast cadence.
  private forceBurst = true;

  readonly performance = new PerformanceManager();
  private callbacks: PipelineCallbacks = {};

  async init(videoElement: HTMLVideoElement, callbacks: PipelineCallbacks = {}) {
    this.video = videoElement;
    this.callbacks = callbacks;

    // 480x360 @ 10 fps preserves crisp lip/iris landmarks while cutting webcam decode overhead by ~33%
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 480 },
        height: { ideal: 360 },
        frameRate: { ideal: 10, max: 10 },
        facingMode: 'user',
      },
      audio: false,
    });

    this.video.srcObject = this.stream;
    await this.video.play();

    this.faceWorker = new Worker(
      new URL('../workers/face-mesh.worker.ts', import.meta.url),
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
          this.performance.setBurst(this.forceBurst || Boolean(data.mouth?.isActive) || data.faceCount > 1);
          this.callbacks.onFaceResult?.(data);
        } else {
          this.callbacks.onError?.(data.error);
        }
        if (this.isRunning) {
          // Count interval from dispatch so real cadence matches nominal cadence
          const elapsed = performance.now() - this.lastDispatchAt;
          this.scheduleNextTick(Math.max(0, this.performance.getFaceIntervalMs() - elapsed));
        }
      }
    };

    this.faceWorker.postMessage({
      type: 'INIT',
      wasmPath: '/wasm',
      modelPath: '/models/face_landmarker.task',
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
      this.lastDispatchAt = now;
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

  setForceBurst(on: boolean) {
    this.forceBurst = on;
  }

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
