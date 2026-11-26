import { describe, it, expect } from 'vitest';
import * as ort from 'onnxruntime-node';
import path from 'node:path';
import {
  YUNET_INPUT_W,
  YUNET_INPUT_H,
  preprocessYuNetImageData,
  decodeYuNetOutputs,
} from '../src/lib/models/yunet';
import {
  preprocessGazeImageData,
  decodeGazeLogits,
} from '../src/lib/models/mobilegaze';

/**
 * Exact reference implementation of yakhyo/gaze-estimation/onnx_inference.py
 * (preprocess + softmax + decode) for parity verification.
 */
function upstreamPythonReferencePreprocess(rgba: Uint8ClampedArray, w: number, h: number): Float32Array {
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const out = new Float32Array(3 * h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const r = rgba[idx] / 255.0;
      const g = rgba[idx + 1] / 255.0;
      const b = rgba[idx + 2] / 255.0;
      out[0 * h * w + y * w + x] = (r - mean[0]) / std[0];
      out[1 * h * w + y * w + x] = (g - mean[1]) / std[1];
      out[2 * h * w + y * w + x] = (b - mean[2]) / std[2];
    }
  }
  return out;
}

function upstreamPythonReferenceDecode(logits: Float32Array): number {
  let maxVal = -Infinity;
  for (let i = 0; i < 90; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }
  let sumExp = 0;
  const exp = new Float64Array(90);
  for (let i = 0; i < 90; i++) {
    exp[i] = Math.exp(logits[i] - maxVal);
    sumExp += exp[i];
  }
  let deg = 0;
  for (let i = 0; i < 90; i++) {
    deg += i * (exp[i] / sumExp);
  }
  return deg * 4 - 180;
}

describe('Stage A: YuNet + MobileOne-S0 Gaze Parity', () => {
  it('runs YuNet and decodes outputs without shape errors', async () => {
    const yunetPath = path.resolve('public/models/face_detection_yunet_2023mar.onnx');
    const session = await ort.InferenceSession.create(yunetPath);

    // face_detection_yunet_2023mar.onnx has fixed [1, 3, 640, 640] input
    const inW = 640;
    const inH = 640;
    const rgba = new Uint8ClampedArray(inW * inH * 4);
    for (let i = 0; i < rgba.length; i += 4) {
      rgba[i] = 128;
      rgba[i + 1] = 128;
      rgba[i + 2] = 128;
      rgba[i + 3] = 255;
    }

    const inputData = preprocessYuNetImageData(rgba, inW, inH);
    const inputName = session.inputNames[0];
    const tensor = new ort.Tensor('float32', inputData, [1, 3, inH, inW]);
    const outputs = await session.run({ [inputName]: tensor });

    const boxes = decodeYuNetOutputs(outputs as any, 640, 480, 0.6, 0.3, inW, inH, 640, 480);
    expect(Array.isArray(boxes)).toBe(true);
  });

  it('matches upstream Python onnx_inference.py within < 2 degrees across 20 frames', async () => {
    const gazePath = path.resolve('public/models/mobileone_s0_gaze.onnx');
    const session = await ort.InferenceSession.create(gazePath);
    const inputName = session.inputNames[0];
    const outNames = session.outputNames;
    expect(outNames.length).toBe(2);

    const W = 448;
    const H = 448;
    let maxYawDiff = 0;
    let maxPitchDiff = 0;

    for (let frameIdx = 0; frameIdx < 20; frameIdx++) {
      // Deterministic synthetic 448x448 face-like pattern per frame
      const rgba = new Uint8ClampedArray(W * H * 4);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = (y * W + x) * 4;
          rgba[i] = (x + frameIdx * 11) % 256;
          rgba[i + 1] = (y + frameIdx * 7) % 256;
          rgba[i + 2] = ((x ^ y) + frameIdx * 13) % 256;
          rgba[i + 3] = 255;
        }
      }

      // TypeScript pipeline
      const tsInput = preprocessGazeImageData(rgba, W, H);
      const tsOut = await session.run({ [inputName]: new ort.Tensor('float32', tsInput, [1, 3, H, W]) });
      const tsAngles = decodeGazeLogits(
        tsOut[outNames[0]].data as Float32Array,
        tsOut[outNames[1]].data as Float32Array
      );

      // Upstream Python reference pipeline
      const refInput = upstreamPythonReferencePreprocess(rgba, W, H);
      const refOut = await session.run({ [inputName]: new ort.Tensor('float32', refInput, [1, 3, H, W]) });
      const refYaw = upstreamPythonReferenceDecode(refOut[outNames[0]].data as Float32Array);
      const refPitch = upstreamPythonReferenceDecode(refOut[outNames[1]].data as Float32Array);

      const yawDiff = Math.abs(tsAngles.yawDeg - refYaw);
      const pitchDiff = Math.abs(tsAngles.pitchDeg - refPitch);
      maxYawDiff = Math.max(maxYawDiff, yawDiff);
      maxPitchDiff = Math.max(maxPitchDiff, pitchDiff);

      expect(yawDiff).toBeLessThan(2.0);
      expect(pitchDiff).toBeLessThan(2.0);
    }

    console.log(
      `[Stage A Parity] 20 frames -> maxYawDiff=${maxYawDiff.toFixed(6)} deg, maxPitchDiff=${maxPitchDiff.toFixed(6)} deg`
    );
  });
});
