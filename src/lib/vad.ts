import { MicVAD } from '@ricky0123/vad-web';

interface SpeechSegment {
  start: number;
  end: number;
}

export class VADManager {
  private vad: MicVAD | null = null;
  private isSpeechCurrentlyHappening = false;
  private currentSpeechStart: number | null = null;
  private speechSegments: SpeechSegment[] = [];
  private onSpeechUpdateCallback?: (isSpeechTriggered: boolean, speechDurationSec: number) => void;

  async init(onSpeechUpdate?: (isSpeechTriggered: boolean, speechDurationSec: number) => void) {
    this.onSpeechUpdateCallback = onSpeechUpdate;

    try {
      this.vad = await MicVAD.new({
        model: 'v5',
        baseAssetPath: '/wasm/',
        onnxWASMBasePath: '/wasm/',
        onSpeechStart: () => {
          this.isSpeechCurrentlyHappening = true;
          this.currentSpeechStart = Date.now();
        },
        onSpeechEnd: () => {
          this.isSpeechCurrentlyHappening = false;
          if (this.currentSpeechStart !== null) {
            this.speechSegments.push({
              start: this.currentSpeechStart,
              end: Date.now(),
            });
            this.currentSpeechStart = null;
          }
          this.evaluateRollingSpeech();
        },
        onVADMisfire: () => {
          this.isSpeechCurrentlyHappening = false;
          this.currentSpeechStart = null;
        },
      });

      await this.vad.start();
    } catch (err) {
      console.error('Silero VAD failed to initialize:', err);
      throw err;
    }
  }

  /**
   * Calculates total speech duration in seconds within the rolling 60-second window.
   */
  getRollingSpeechDuration(now = Date.now()): number {
    const windowStart = now - 60_000;
    // Evict segments older than 60s
    this.speechSegments = this.speechSegments.filter((seg) => seg.end > windowStart);

    let totalMs = 0;
    for (const seg of this.speechSegments) {
      const segStart = Math.max(seg.start, windowStart);
      const segEnd = Math.min(seg.end, now);
      if (segEnd > segStart) {
        totalMs += segEnd - segStart;
      }
    }

    // Include currently ongoing speech
    if (this.isSpeechCurrentlyHappening && this.currentSpeechStart !== null) {
      const liveStart = Math.max(this.currentSpeechStart, windowStart);
      if (now > liveStart) {
        totalMs += now - liveStart;
      }
    }

    return totalMs / 1000.0;
  }

  evaluateRollingSpeech(now = Date.now()): { isTriggered: boolean; durationSec: number } {
    const durationSec = this.getRollingSpeechDuration(now);
    const isTriggered = durationSec >= 8.0; // >= 8s in 60s window
    this.onSpeechUpdateCallback?.(isTriggered, durationSec);
    return { isTriggered, durationSec };
  }

  isSpeakingNow(): boolean {
    return this.isSpeechCurrentlyHappening;
  }

  pause(): void {
    this.vad?.pause();
  }

  start(): void {
    this.vad?.start();
  }

  destroy(): void {
    this.vad?.pause();
    this.vad = null;
  }
}
