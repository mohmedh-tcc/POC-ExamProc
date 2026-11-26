export interface FaceBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
}

export const YUNET_INPUT_W = 320;
export const YUNET_INPUT_H = 256; // Multiple of 32 (240 active + 16 bottom padding)
export const YUNET_ACTIVE_H = 240;

const STRIDES = [8, 16, 32] as const;

/**
 * Converts RGBA ImageData (320x256) into BGR Float32Array NCHW [1, 3, 256, 320] in [0, 255].
 */
export function preprocessYuNetImageData(
  rgba: Uint8ClampedArray,
  width = YUNET_INPUT_W,
  height = YUNET_INPUT_H
): Float32Array {
  const hw = width * height;
  const out = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    const p = i * 4;
    out[i] = rgba[p + 2];          // B
    out[hw + i] = rgba[p + 1];     // G
    out[2 * hw + i] = rgba[p];     // R
  }
  return out;
}

function iou(a: FaceBox, b: FaceBox): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

function nms(boxes: FaceBox[], iouThreshold = 0.3): FaceBox[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept: FaceBox[] = [];
  for (const box of sorted) {
    let suppressed = false;
    for (const k of kept) {
      if (iou(box, k) > iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push(box);
  }
  return kept;
}

/**
 * Decodes YuNet 12-head output tensors into scaled FaceBox[] in source frame coordinates.
 */
export function decodeYuNetOutputs(
  outputs: Record<string, { data: Float32Array | number[] }>,
  srcW: number,
  srcH: number,
  scoreThreshold = 0.6,
  iouThreshold = 0.3,
  inputW = YUNET_INPUT_W,
  inputH = YUNET_INPUT_H,
  activeW = YUNET_INPUT_W,
  activeH = YUNET_ACTIVE_H
): FaceBox[] {
  const candidates: FaceBox[] = [];

  for (const stride of STRIDES) {
    const cls = outputs[`cls_${stride}`]?.data;
    const obj = outputs[`obj_${stride}`]?.data;
    const bbox = outputs[`bbox_${stride}`]?.data;
    if (!cls || !obj || !bbox) continue;

    const cols = Math.floor(inputW / stride);
    const rows = Math.floor(inputH / stride);
    const count = cols * rows;

    for (let idx = 0; idx < count; idx++) {
      const c = Math.max(0, Math.min(1, cls[idx]));
      const o = Math.max(0, Math.min(1, obj[idx]));
      const score = Math.sqrt(c * o);
      if (score < scoreThreshold) continue;

      const row = Math.floor(idx / cols);
      const col = idx % cols;
      const bOff = idx * 4;

      const cx = (col + bbox[bOff]) * stride;
      const cy = (row + bbox[bOff + 1]) * stride;
      const w = Math.exp(bbox[bOff + 2]) * stride;
      const h = Math.exp(bbox[bOff + 3]) * stride;

      candidates.push({
        x1: cx - w / 2,
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
        score,
      });
    }
  }

  const filtered = nms(candidates, iouThreshold);
  const scaleX = srcW / activeW;
  const scaleY = srcH / activeH;

  return filtered.map((b) => ({
    x1: Math.max(0, Math.min(srcW, b.x1 * scaleX)),
    y1: Math.max(0, Math.min(srcH, b.y1 * scaleY)),
    x2: Math.max(0, Math.min(srcW, b.x2 * scaleX)),
    y2: Math.max(0, Math.min(srcH, b.y2 * scaleY)),
    score: b.score,
  }));
}

/**
 * Selects the primary face (largest bounding box area).
 */
export function selectPrimaryFace(boxes: FaceBox[]): FaceBox | null {
  if (boxes.length === 0) return null;
  let best = boxes[0];
  let bestArea = (best.x2 - best.x1) * (best.y2 - best.y1);
  for (let i = 1; i < boxes.length; i++) {
    const b = boxes[i];
    const area = (b.x2 - b.x1) * (b.y2 - b.y1);
    if (area > bestArea) {
      best = b;
      bestArea = area;
    }
  }
  return best;
}
