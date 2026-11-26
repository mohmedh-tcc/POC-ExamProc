/**
 * Rolling window of boolean samples measured in time, not frames, so thresholds
 * hold across 2 Hz idle, 8 Hz burst, and latency throttling.
 */
export class RollingFlags {
  private buf: Array<{ t: number; v: boolean }> = [];

  constructor(private readonly spanMs: number) {}

  push(v: boolean, now: number): void {
    this.buf.push({ t: now, v });
    while (this.buf.length > 0 && this.buf[0].t < now - this.spanMs) {
      this.buf.shift();
    }
  }

  /** Share of true samples, or null until the kept samples span at least `minCoverMs`. */
  fraction(minCoverMs: number): number | null {
    const n = this.buf.length;
    if (n < 2 || this.buf[n - 1].t - this.buf[0].t < minCoverMs) return null;
    let trues = 0;
    for (const s of this.buf) {
      if (s.v) trues++;
    }
    return trues / n;
  }
}
