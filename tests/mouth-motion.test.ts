import { describe, it, expect } from 'vitest';
import { MouthMotionTracker } from '../src/lib/math/mouth-motion';
import { gauss, mouthLandmarks, mulberry32, speechLikeMar } from './_mouth-fixtures';

const W = 480;
const H = 360;

function run(signal: (tS: number) => number, hz: number, secs: number, noise = 0, seed = 1) {
  const tracker = new MouthMotionTracker();
  const rand = mulberry32(seed);
  let n = 0;
  let speaking = 0;
  let active = 0;
  for (let t = 0; t <= secs * 1000; t += 1000 / hz) {
    const mar = Math.max(0, signal(t / 1000) + noise * gauss(rand));
    const r = tracker.update(mouthLandmarks(mar, W, H), W, H, t);
    if (t >= 3000) {
      n++;
      if (r.isSpeaking) speaking++;
      if (r.isActive) active++;
    }
  }
  return { speaking: speaking / n, active: active / n };
}

const SEEDS = [1, 2, 3, 4, 5];

describe('MouthMotionTracker', () => {
  it('reports the same MAR for the same physical mouth on any frame aspect', () => {
    for (const [w, h] of [
      [480, 360],
      [640, 480],
      [640, 360],
      [1280, 720],
    ]) {
      const r = new MouthMotionTracker().update(mouthLandmarks(0.2, w, h), w, h, 0);
      expect(r.mar).toBeCloseTo(0.2, 6);
    }
  });

  it('compensates horizontal mouth foreshortening via cos(headYawRad)', () => {
    const yawRad = Math.PI / 6; // 30 deg turn shrinks projected width by cos(30 deg), inflating raw MAR by 1/cos(30 deg)
    const lm = mouthLandmarks(0.2 / Math.cos(yawRad), W, H, 60 * Math.cos(yawRad));
    const r = new MouthMotionTracker().update(lm, W, H, 0, yawRad);
    expect(r.mar).toBeCloseTo(0.2, 5);
  });

  it('flags speech-like lip motion sampled at the 8 Hz burst rate', () => {
    for (const seed of SEEDS) {
      expect(run(speechLikeMar(30, seed), 8, 30, 0.005, seed).speaking).toBeGreaterThan(0.9);
    }
  });

  it('gives the same verdict at 8 and 10 Hz (window is in time, not frames)', () => {
    const sig = speechLikeMar(30, 2);
    expect(run(sig, 8, 30, 0.005, 2).speaking).toBeGreaterThan(0.9);
    expect(run(sig, 10, 30, 0.005, 2).speaking).toBeGreaterThan(0.9);
  });

  it('never calls speech from 2 Hz idle samples, but wakes the burst', () => {
    for (const seed of SEEDS) {
      const r = run(speechLikeMar(30, seed), 2, 30, 0.005, seed);
      expect(r.speaking).toBe(0);
      expect(r.active).toBeGreaterThan(0.85);
    }
  });

  it('stays quiet and idle on a still mouth with landmark jitter', () => {
    for (const seed of SEEDS) {
      const r = run(() => 0.03, 8, 60, 0.005, seed);
      expect(r.speaking).toBe(0);
      expect(r.active).toBe(0);
    }
  });

  it('does not call speech from heavy jitter (sigma = 0.012)', () => {
    for (const seed of SEEDS) {
      expect(run(() => 0.03, 8, 60, 0.012, seed).speaking).toBeLessThan(0.05);
    }
  });

  it('ignores a held-open mouth', () => {
    const r = run(() => 0.25, 8, 20);
    expect(r.speaking).toBe(0);
    expect(r.active).toBe(0);
  });

  it('wakes the burst for a yawn but does not call it speech', () => {
    const yawn = (t: number) =>
      0.03 + 0.3 * (t % 12 < 3 ? Math.max(0, Math.sin(Math.PI * ((t % 12) / 3))) : 0);
    const r = run(yawn, 8, 60);
    expect(r.speaking).toBe(0);
    expect(r.active).toBeGreaterThan(0.2);
  });

  it('reset() drops the history', () => {
    const tracker = new MouthMotionTracker();
    const sig = speechLikeMar(10, 1);
    for (let t = 0; t <= 3000; t += 125) {
      tracker.update(mouthLandmarks(sig(t / 1000), W, H), W, H, t);
    }
    tracker.reset();
    expect(tracker.update(mouthLandmarks(0.03, W, H), W, H, 3125).isActive).toBe(false);
  });
});
