import { describe, it, expect } from 'vitest';
import { PerformanceManager } from '../src/lib/performance';

describe('PerformanceManager cadence', () => {
  it('idles at 2 Hz and bursts at 8 Hz', () => {
    const p = new PerformanceManager();
    expect(p.getFaceIntervalMs()).toBe(500);
    p.setBurst(true);
    expect(p.getFaceIntervalMs()).toBe(125);
    p.setBurst(false);
    expect(p.getFaceIntervalMs()).toBe(500);
  });

  it('stretches both cadences when P95 latency blows the budget, and recovers', () => {
    const p = new PerformanceManager();
    for (let i = 0; i < 10; i++) p.recordFaceLatency(200);
    expect(p.getFaceIntervalMs()).toBe(667);
    p.setBurst(true);
    expect(p.getFaceIntervalMs()).toBe(167);
    for (let i = 0; i < 10; i++) p.recordFaceLatency(200);
    expect(p.getFaceIntervalMs()).toBe(250);
    for (let i = 0; i < 40; i++) p.recordFaceLatency(30);
    expect(p.getFaceIntervalMs()).toBe(125);
  });
});
