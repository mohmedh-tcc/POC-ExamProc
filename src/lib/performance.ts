export class PerformanceManager {
  private faceLatencies: number[] = [];
  private objectLatencies: number[] = [];
  private readonly maxWindow = 30;

  // Face rate steps: 5 Hz (200ms) -> 3 Hz (333ms) -> 2 Hz (500ms)
  private readonly faceRates = [200, 333, 500];
  private faceRateIndex = 0;

  // Object interval steps: 2.5s (2500ms) -> 4.0s (4000ms) -> 6.0s (6000ms)
  private readonly objectIntervals = [2500, 4000, 6000];
  private objectIntervalIndex = 0;

  private isDegraded = false;

  recordFaceLatency(ms: number): void {
    this.faceLatencies.push(ms);
    if (this.faceLatencies.length > this.maxWindow) {
      this.faceLatencies.shift();
    }
    this.evaluateThrottling();
  }

  recordObjectLatency(ms: number): void {
    this.objectLatencies.push(ms);
    if (this.objectLatencies.length > this.maxWindow) {
      this.objectLatencies.shift();
    }
    this.evaluateThrottling();
  }

  private calculateP95(samples: number[]): number {
    if (samples.length === 0) return 0;
    const sorted = [...samples].sort((a, b) => a - b);
    const index = Math.floor(0.95 * (sorted.length - 1));
    return sorted[index];
  }

  getFaceP95(): number {
    return this.calculateP95(this.faceLatencies);
  }

  getObjectP95(): number {
    return this.calculateP95(this.objectLatencies);
  }

  private evaluateThrottling(): void {
    if (this.faceLatencies.length >= 10) {
      const p95 = this.getFaceP95();
      // Budget: 120 ms
      if (p95 > 120 && this.faceRateIndex < this.faceRates.length - 1) {
        this.faceRateIndex += 1;
        console.warn(`Face Landmarker P95 (${p95.toFixed(1)}ms) exceeded budget. Throttling interval to ${this.faceRates[this.faceRateIndex]}ms`);
      } else if (p95 < 70 && this.faceRateIndex > 0) {
        // Step back up if performance recovered
        this.faceRateIndex -= 1;
      }
    }

    if (this.objectLatencies.length >= 5) {
      const p95 = this.getObjectP95();
      if (p95 > 1500 && this.objectIntervalIndex < this.objectIntervals.length - 1) {
        this.objectIntervalIndex += 1;
        console.warn(`Object Detector P95 (${p95.toFixed(1)}ms) exceeded budget. Throttling interval to ${this.objectIntervals[this.objectIntervalIndex]}ms`);
      } else if (p95 < 600 && this.objectIntervalIndex > 0) {
        this.objectIntervalIndex -= 1;
      }
    }

    // Degraded coverage check: if at lowest rate and still over budget
    const faceAtFloor = this.faceRateIndex === this.faceRates.length - 1;
    const faceOverBudget = this.getFaceP95() > 120;
    this.isDegraded = faceAtFloor && faceOverBudget;
  }

  getFaceIntervalMs(): number {
    return this.faceRates[this.faceRateIndex];
  }

  getObjectIntervalMs(): number {
    return this.objectIntervals[this.objectIntervalIndex];
  }

  getCurrentFaceFps(): number {
    return Number((1000 / this.getFaceIntervalMs()).toFixed(1));
  }

  isDegradedCoverage(): boolean {
    return this.isDegraded;
  }

  getStats() {
    return {
      faceP95: this.getFaceP95(),
      objectP95: this.getObjectP95(),
      faceIntervalMs: this.getFaceIntervalMs(),
      objectIntervalMs: this.getObjectIntervalMs(),
      faceFps: this.getCurrentFaceFps(),
      isDegraded: this.isDegraded,
    };
  }
}
