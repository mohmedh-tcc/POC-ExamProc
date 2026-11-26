import { describe, it, expect } from 'vitest';
import { RollingFlags } from '../src/lib/engine/rolling-window';

describe('RollingFlags', () => {
  it('is null until the samples span the minimum coverage', () => {
    const w = new RollingFlags(3500);
    w.push(true, 0);
    expect(w.fraction(2000)).toBeNull();
    w.push(true, 1000);
    expect(w.fraction(2000)).toBeNull();
    w.push(true, 2000);
    expect(w.fraction(2000)).toBe(1);
  });

  it('forgets samples older than the span', () => {
    const w = new RollingFlags(3500);
    for (let t = 0; t <= 3000; t += 500) w.push(true, t);
    for (let t = 3500; t <= 7000; t += 500) w.push(false, t);
    expect(w.fraction(2000)).toBe(0);
  });

  it('gives the same verdict at 2 Hz and 8 Hz for the same 3 s of no-face', () => {
    for (const stepMs of [500, 125]) {
      const w = new RollingFlags(3500);
      for (let t = 0; t <= 3000; t += stepMs) w.push(true, t);
      expect((w.fraction(2000) ?? 0) >= 0.9).toBe(true);
    }
  });

  it('does not confirm a second face from a single idle-rate sample', () => {
    const w = new RollingFlags(1250);
    w.push(false, 0);
    w.push(false, 500);
    w.push(true, 1000);
    expect((w.fraction(750) ?? 0) >= 0.4).toBe(false);
    w.push(true, 1500);
    expect((w.fraction(750) ?? 0) >= 0.4).toBe(true);
  });
});
