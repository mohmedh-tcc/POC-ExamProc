import { FaceBox } from './yunet';

const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
const IMAGENET_STD = [0.229, 0.224, 0.225] as const;
const BINS = 90;
const BIN_WIDTH = 4;
const ANGLE_OFFSET = 180;

export interface GazeAngles {
  yawDeg: number;
  pitchDeg: number;
}

/**
 * Computes the face crop rectangle with optional margin (default 0% to match upstream onnx_inference.py).
 */
export function computeCropRect(
  box: FaceBox,
  srcW: number,
  srcH: number,
  margin = 0
): { sx: number; sy: number; sw: number; sh: number } {
  const bw = Math.max(1, box.x2 - box.x1);
  const bh = Math.max(1, box.y2 - box.y1);
  const padX = bw * margin;
  const padY = bh * margin;

  const x1 = Math.max(0, Math.floor(box.x1 - padX));
  const y1 = Math.max(0, Math.floor(box.y1 - padY));
  const x2 = Math.min(srcW, Math.ceil(box.x2 + padX));
  const y2 = Math.min(srcH, Math.ceil(box.y2 + padY));

  return {
    sx: x1,
    sy: y1,
    sw: Math.max(1, x2 - x1),
    sh: Math.max(1, y2 - y1),
  };
}

/**
 * Converts RGBA ImageData (targetW x targetH) to RGB Float32Array NCHW [1, 3, targetH, targetW]
 * with ImageNet normalization, matching yakhyo/gaze-estimation/onnx_inference.py.
 */
export function preprocessGazeImageData(
  rgba: Uint8ClampedArray,
  width: number,
  height: number
): Float32Array {
  const hw = width * height;
  const out = new Float32Array(3 * hw);

  for (let i = 0; i < hw; i++) {
    const p = i * 4;
    out[i] = (rgba[p] / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];             // R
    out[hw + i] = (rgba[p + 1] / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];    // G
    out[2 * hw + i] = (rgba[p + 2] / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];// B
  }

  return out;
}

function softmaxDecodeDeg(logits: Float32Array | number[]): number {
  let maxLogit = -Infinity;
  for (let i = 0; i < BINS; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  let sumExp = 0;
  const probs = new Float32Array(BINS);
  for (let i = 0; i < BINS; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    probs[i] = e;
    sumExp += e;
  }

  let weightedSum = 0;
  for (let i = 0; i < BINS; i++) {
    weightedSum += i * (probs[i] / sumExp);
  }

  return weightedSum * BIN_WIDTH - ANGLE_OFFSET;
}

/**
 * Decodes MobileOne-S0 Gaze [1, 90] yaw & pitch logits into degrees in [-180, 180].
 * Upstream tensor order: outputs[0] = yaw, outputs[1] = pitch.
 */
export function decodeGazeLogits(
  yawLogits: Float32Array | number[],
  pitchLogits: Float32Array | number[]
): GazeAngles {
  return {
    yawDeg: softmaxDecodeDeg(yawLogits),
    pitchDeg: softmaxDecodeDeg(pitchLogits),
  };
}
