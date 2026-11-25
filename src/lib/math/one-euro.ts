/**
 * One-Euro Filter for real-time low-pass filtering of noisy signals with adaptive cutoff.
 * Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Human-Computer Interaction", CHI 2012.
 */
class LowPassFilter {
  private y: number | null = null;
  private s: number | null = null;

  filter(value: number, alpha: number): number {
    if (this.y === null) {
      this.s = value;
      this.y = value;
      return value;
    }
    this.y = value;
    this.s = alpha * value + (1.0 - alpha) * (this.s as number);
    return this.s;
  }

  hasLastRawValue(): boolean {
    return this.y !== null;
  }

  lastRawValue(): number {
    return this.y ?? 0;
  }
}

export class OneEuroFilter {
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;
  private xFilter: LowPassFilter;
  private dxFilter: LowPassFilter;
  private lastTime: number | null = null;

  /**
   * @param minCutoff Minimum cutoff frequency in Hz (default: 1.0) - lower values remove more jitter when stationary
   * @param beta Speed coefficient (default: 0.007) - higher values reduce lag during rapid movements
   * @param dCutoff Cutoff frequency for derivative in Hz (default: 1.0)
   */
  constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xFilter = new LowPassFilter();
    this.dxFilter = new LowPassFilter();
  }

  private alpha(cutoff: number, dt: number): number {
    const te = dt;
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }

  filter(value: number, timestamp: number): number {
    if (this.lastTime === null) {
      this.lastTime = timestamp;
      return this.xFilter.filter(value, 1.0);
    }

    const dt = Math.max((timestamp - this.lastTime) / 1000.0, 1e-4);
    this.lastTime = timestamp;

    const dx = (value - this.xFilter.lastRawValue()) / dt;
    const edx = this.dxFilter.filter(dx, this.alpha(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);

    return this.xFilter.filter(value, this.alpha(cutoff, dt));
  }

  reset(): void {
    this.lastTime = null;
    this.xFilter = new LowPassFilter();
    this.dxFilter = new LowPassFilter();
  }
}
