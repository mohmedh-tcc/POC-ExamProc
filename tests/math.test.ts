import { describe, it, expect } from 'vitest';
import { decomposeTransformMatrix } from '../src/lib/math/transform-matrix';
import { computeIrisGaze, NormalizedLandmark } from '../src/lib/math/iris-gaze';
import { OneEuroFilter } from '../src/lib/math/one-euro';

describe('Transform Matrix Euler Decomposition', () => {
  it('decomposes identity matrix to zero angles', () => {
    const identity = [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ];
    const angles = decomposeTransformMatrix(identity);
    expect(angles.pitch).toBeCloseTo(0, 3);
    expect(angles.yaw).toBeCloseTo(0, 3);
    expect(angles.roll).toBeCloseTo(0, 3);
  });

  it('clamps and orthonormalizes -R21 to prevent NaN when floating-point precision slightly overshoots', () => {
    const matrixWithOvershoot = [
      1, 0, 0, 0,
      0, 1, -1.0000002, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ];
    const angles = decomposeTransformMatrix(matrixWithOvershoot);
    expect(isNaN(angles.pitch)).toBe(false);
    expect(Math.abs(angles.pitch)).toBeLessThanOrEqual(90);
  });

  it('correctly calculates pitch when looking down', () => {
    const rad = (30 * Math.PI) / 180;
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    
    const pitchMatrix = [
      1, 0, 0, 0,
      0, c, s, 0,
      0, -s, c, 0,
      0, 0, 0, 1
    ];
    const angles = decomposeTransformMatrix(pitchMatrix);
    expect(angles.pitch).toBeCloseTo(-30, 1);
  });
});

describe('Iris Gaze Calculation', () => {
  it('correctly calculates centered iris ratios', () => {
    const landmarks: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));

    // Right eye: Outer 33, Inner 133, Upper 159, Lower 145, Iris 468
    landmarks[33] = { x: 0.3, y: 0.5, z: 0 };
    landmarks[133] = { x: 0.5, y: 0.5, z: 0 };
    landmarks[159] = { x: 0.4, y: 0.45, z: 0 };
    landmarks[145] = { x: 0.4, y: 0.55, z: 0 };
    landmarks[468] = { x: 0.4, y: 0.5, z: 0 }; // exactly centered

    // Left eye: Inner 362, Outer 263, Upper 386, Lower 374, Iris 473
    landmarks[362] = { x: 0.6, y: 0.5, z: 0 };
    landmarks[263] = { x: 0.8, y: 0.5, z: 0 };
    landmarks[386] = { x: 0.7, y: 0.45, z: 0 };
    landmarks[374] = { x: 0.7, y: 0.55, z: 0 };
    landmarks[473] = { x: 0.7, y: 0.5, z: 0 }; // exactly centered

    const result = computeIrisGaze(landmarks);
    expect(result.isBlinking).toBe(false);
    expect(result.hRatio).toBeCloseTo(0.5, 2);
    expect(result.vRatio).toBeCloseTo(0.5, 2);
  });

  it('detects blink and preserves previous gaze ratio without dividing by near zero', () => {
    const landmarks: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));

    landmarks[33] = { x: 0.3, y: 0.5, z: 0 };
    landmarks[133] = { x: 0.5, y: 0.5, z: 0 };
    landmarks[159] = { x: 0.4, y: 0.501, z: 0 };
    landmarks[145] = { x: 0.4, y: 0.500, z: 0 };
    landmarks[468] = { x: 0.4, y: 0.500, z: 0 };

    landmarks[362] = { x: 0.6, y: 0.5, z: 0 };
    landmarks[263] = { x: 0.8, y: 0.5, z: 0 };
    landmarks[386] = { x: 0.7, y: 0.501, z: 0 };
    landmarks[374] = { x: 0.7, y: 0.500, z: 0 };
    landmarks[473] = { x: 0.7, y: 0.500, z: 0 };

    const previousGaze = {
      hRatio: 0.35,
      vRatio: 0.65,
      isBlinking: false,
      rightOpenness: 0.3,
      leftOpenness: 0.3,
      dx: -0.06,
      dy: 0.05,
      eyeYawRad: -0.15,
      eyePitchRad: -0.14,
      eyeVector: [0.15, -0.14, 0.98] as [number, number, number],
      confidence: 1,
    };

    const result = computeIrisGaze(landmarks, previousGaze);
    expect(result.isBlinking).toBe(true);
    expect(result.hRatio).toBe(0.35);
    expect(result.vRatio).toBe(0.65);
  });

  it('remains invariant to head roll (tilt) via eye-local basis projection', () => {
    const landmarks: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));

    // Rotate eye geometry by 25 degrees around center (0.4, 0.5)
    const theta = (25 * Math.PI) / 180;
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const rot = (x: number, y: number, cx: number, cy: number) => ({
      x: cx + (x - cx) * cos - (y - cy) * sin,
      y: cy + (x - cx) * sin + (y - cy) * cos,
      z: 0,
    });

    landmarks[33] = rot(0.3, 0.5, 0.4, 0.5);
    landmarks[133] = rot(0.5, 0.5, 0.4, 0.5);
    landmarks[159] = rot(0.4, 0.45, 0.4, 0.5);
    landmarks[145] = rot(0.4, 0.55, 0.4, 0.5);
    landmarks[468] = rot(0.4, 0.5, 0.4, 0.5); // Centered iris

    landmarks[362] = rot(0.6, 0.5, 0.7, 0.5);
    landmarks[263] = rot(0.8, 0.5, 0.7, 0.5);
    landmarks[386] = rot(0.7, 0.45, 0.7, 0.5);
    landmarks[374] = rot(0.7, 0.55, 0.7, 0.5);
    landmarks[473] = rot(0.7, 0.5, 0.7, 0.5); // Centered iris

    const result = computeIrisGaze(landmarks);
    expect(result.isBlinking).toBe(false);
    expect(result.hRatio).toBeCloseTo(0.5, 2);
    expect(result.vRatio).toBeCloseTo(0.5, 2);
  });
});

describe('OneEuroFilter', () => {
  it('dampens rapid alternating noise after initial warm-up', () => {
    const filter = new OneEuroFilter(1.0, 0.007);
    let time = 1000;
    const baseValue = 25.0;

    // Warm-up filter
    filter.filter(baseValue, time);

    // Simulate 30 frames with +- 1.5 degree jitter
    for (let i = 0; i < 30; i++) {
      time += 200; // 5Hz (200ms per frame)
      const noise = (i % 2 === 0 ? 1 : -1) * 1.5;
      const filtered = filter.filter(baseValue + noise, time);
      // The filtered value should be dampened compared to the 1.5 deg noise
      expect(Math.abs(filtered - baseValue)).toBeLessThan(1.2);
    }
  });
});
