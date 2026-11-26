import { SignalFamily } from './state-machine';

export type RiskLevel = 'green' | 'amber' | 'red';

export class LeakyBucketEngine {
  private currentScore = 0;
  private lastUpdateTime: number;
  private enabled = true;
  private readonly halfLifeMs = 120_000; // 2 minutes (120 seconds)
  private readonly lambda = Math.LN2 / 120_000; // decay constant per ms

  // Tracks the most recent event timestamp for each family
  private familyLastActiveTimes: Record<SignalFamily, number | null> = {
    vision: null,
    audio: null,
    browser: null,
  };

  constructor(initialTime = Date.now()) {
    this.lastUpdateTime = initialTime;
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  setEnabled(enabled: boolean, now = Date.now()): void {
    if (this.enabled) {
      this.decay(now);
    } else {
      this.lastUpdateTime = now;
    }
    this.enabled = enabled;
  }

  /**
   * Applies continuous exponential decay up to timestamp `now` when leaky bucket is enabled.
   */
  private decay(now: number): void {
    if (now <= this.lastUpdateTime) return;
    if (this.enabled) {
      const dt = now - this.lastUpdateTime;
      this.currentScore = this.currentScore * Math.exp(-this.lambda * dt);
    }
    this.lastUpdateTime = now;
  }

  /**
   * Adds an event weight to the bucket at time `now`.
   * When enabled: applies 1.5x multiplier if a different family was active within 30s.
   * When disabled: no decay, no multiplier, and reduces flag points by 50% (0.5x).
   */
  addWeight(weight: number, family: SignalFamily, now: number): number {
    this.decay(now);

    if (!this.enabled) {
      const added = weight * 0.5;
      this.currentScore += added;
      this.familyLastActiveTimes[family] = now;
      return added;
    }

    let multiplier = 1.0;
    const thirtySec = 30_000;
    for (const [f, lastTime] of Object.entries(this.familyLastActiveTimes) as [SignalFamily, number | null][]) {
      if (f !== family && lastTime !== null && now - lastTime <= thirtySec) {
        multiplier = 1.5;
        break;
      }
    }

    const added = weight * multiplier;
    this.currentScore += added;
    this.familyLastActiveTimes[family] = now;

    return added;
  }

  getScore(now: number): number {
    this.decay(now);
    return Math.max(0, this.currentScore);
  }

  getLevel(now: number): RiskLevel {
    const score = this.getScore(now);
    if (score < 4.0) return 'green';
    if (score < 8.0) return 'amber';
    return 'red';
  }

  reset(now = Date.now()): void {
    this.currentScore = 0;
    this.lastUpdateTime = now;
    this.familyLastActiveTimes = {
      vision: null,
      audio: null,
      browser: null,
    };
  }
}
