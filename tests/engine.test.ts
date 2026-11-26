import { describe, it, expect } from 'vitest';
import { buildCalibrationProfile } from '../src/lib/engine/calibrator';
import { DebounceStateMachine } from '../src/lib/engine/state-machine';
import { LeakyBucketEngine } from '../src/lib/engine/leaky-bucket';

describe('Calibration Engine', () => {
  it('correctly establishes trimmed personal bounding envelopes with margins', () => {
    // 10 samples ranging from yaw -10 to +10, pitch -5 to +15
    const samples = [
      { yaw: -10, pitch: -5, hRatio: 0.3, vRatio: 0.3 },
      { yaw: -8, pitch: -3, hRatio: 0.35, vRatio: 0.35 },
      { yaw: -5, pitch: 0, hRatio: 0.4, vRatio: 0.4 },
      { yaw: 0, pitch: 5, hRatio: 0.5, vRatio: 0.5 },
      { yaw: 2, pitch: 8, hRatio: 0.55, vRatio: 0.55 },
      { yaw: 5, pitch: 10, hRatio: 0.6, vRatio: 0.6 },
      { yaw: 8, pitch: 12, hRatio: 0.65, vRatio: 0.65 },
      { yaw: 10, pitch: 15, hRatio: 0.7, vRatio: 0.7 },
    ];

    const profile = buildCalibrationProfile(samples);

    // Wide anatomical yaw bounds and trimmed pitch floor
    expect(profile.yawMin).toBeLessThanOrEqual(-10);
    expect(profile.yawMax).toBeGreaterThanOrEqual(10);
    expect(profile.pitchMin).toBeLessThanOrEqual(-5);
    expect(profile.pitchMax).toBeGreaterThanOrEqual(15);
  });
});

describe('Debounce State Machine', () => {
  it('transitions from IDLE to ACTIVE after onset and honors 2s release hysteresis', () => {
    const sm = new DebounceStateMachine('LOOKING_DOWN'); // base weight 2, 3s onset
    let now = 1000;

    // 1. Condition turns true -> remains IDLE until 3s onset elapses
    let res = sm.update(true, now);
    expect(res.state).toBe('IDLE');
    expect(res.newActiveEvent).toBe(false);

    now += 2000; // 2s elapsed (< 3s)
    res = sm.update(true, now);
    expect(res.state).toBe('IDLE');

    now += 1000; // 3s elapsed -> transitions to ACTIVE
    res = sm.update(true, now);
    expect(res.state).toBe('ACTIVE');
    expect(res.newActiveEvent).toBe(true);

    // 2. Condition temporarily false for 1s -> RELEASED (not IDLE yet)
    now += 1000;
    res = sm.update(false, now);
    expect(res.state).toBe('RELEASED');
    expect(res.releasedEvent).toBe(false);

    // 3. Condition becomes true again before 2s -> returns to ACTIVE without a new event
    now += 500;
    res = sm.update(true, now);
    expect(res.state).toBe('ACTIVE');
    expect(res.newActiveEvent).toBe(false);

    // 4. Condition false for > 2s -> transitions to IDLE
    now += 1000;
    res = sm.update(false, now); // enters RELEASED
    expect(res.state).toBe('RELEASED');

    now += 2100;
    res = sm.update(false, now); // 2.1s later
    expect(res.state).toBe('IDLE');
    expect(res.releasedEvent).toBe(true);
  });

  it('calculates sustained bonus (+0.25 * weight every 10s)', () => {
    const sm = new DebounceStateMachine('FROZEN_FEED', 0); // weight 5, 0ms onset for test
    let now = 1000;

    // Starts active
    sm.update(true, now);

    // After 9s, no bonus yet
    now += 9000;
    let res = sm.update(true, now);
    expect(res.sustainedBonus).toBe(0);

    // After 11s total (crosses 10s boundary), earns 0.25 * 5 = 1.25
    now += 2000;
    res = sm.update(true, now);
    expect(res.sustainedBonus).toBeCloseTo(1.25, 2);

    // After 21s total, earns another 1.25
    now += 10000;
    res = sm.update(true, now);
    expect(res.sustainedBonus).toBeCloseTo(1.25, 2);
  });

  it('triggers +2 pattern bonus on 5th episode of HEAD_OFF_SCREEN within 5 minutes', () => {
    const sm = new DebounceStateMachine('HEAD_OFF_SCREEN'); // 4s onset
    let now = 10_000;

    // Trigger 4 episodes and release each
    for (let i = 0; i < 4; i++) {
      sm.update(true, now);
      now += 4000; // 4s onset
      sm.update(true, now);
      now += 1000;
      sm.update(false, now);
      now += 2500; // release
      sm.update(false, now);
    }

    // 5th episode triggers within the same 5-minute window
    now += 1000;
    sm.update(true, now);
    now += 4000;
    const res = sm.update(true, now);
    expect(res.newActiveEvent).toBe(true);
    expect(res.patternBonus).toBe(2);
  });
});

describe('Leaky Bucket Risk Engine', () => {
  it('decays with exact 120s half-life', () => {
    const t0 = 100_000;
    const lb = new LeakyBucketEngine(t0);

    // Add 10 points
    lb.addWeight(10, 'vision', t0);
    expect(lb.getScore(t0)).toBeCloseTo(10, 2);

    // After 120s (120,000 ms), score should be exactly half (5.0)
    const t120 = t0 + 120_000;
    expect(lb.getScore(t120)).toBeCloseTo(5.0, 1);

    // After 240s, score should be 2.5
    const t240 = t0 + 240_000;
    expect(lb.getScore(t240)).toBeCloseTo(2.5, 1);
  });

  it('applies 1.5x cross-family multiplier when different family triggers within 30s', () => {
    const t0 = 100_000;
    const lb = new LeakyBucketEngine(t0);

    // Vision event (weight 4)
    const addedVision = lb.addWeight(4, 'vision', t0);
    expect(addedVision).toBe(4.0);

    // Audio event 15s later (weight 2): within 30s of vision -> 1.5x multiplier -> 3.0 added
    const t15 = t0 + 15_000;
    const addedAudio = lb.addWeight(2, 'audio', t15);
    expect(addedAudio).toBe(3.0);
  });

  it('maps scores to Green, Amber, Red levels correctly', () => {
    const t0 = 100_000;
    const lb = new LeakyBucketEngine(t0);

    expect(lb.getLevel(t0)).toBe('green');

    // Add 5 points -> Amber (4 <= score < 8)
    lb.addWeight(5, 'vision', t0);
    expect(lb.getLevel(t0)).toBe('amber');

    // Add 4 more points -> Total 9 -> Red (>= 8)
    lb.addWeight(4, 'vision', t0);
    expect(lb.getLevel(t0)).toBe('red');
  });
});
