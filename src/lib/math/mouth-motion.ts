import { NormalizedLandmark } from './iris-gaze';

export interface MouthMotion {
  mar: number;
  isSpeaking: boolean;
  /** Open/close transitions counted in the window. */
  motionScore: number;
  /** Lips moved enough lately that the caller should sample faster to resolve speech. */
  isActive: boolean;
}

// Time window so the verdict is consistent across 2 Hz idle, 8 Hz burst, and throttling.
const WINDOW_MS = 2500;
const STALE_GAP_MS = 1500;
// Hysteresis and minimum opening in aspect- & yaw-corrected MAR.
const HYST = 0.034;
const MIN_OPEN = 0.045;
const MIN_PEAK_TO_PEAK = 0.035;
// ~4 samples/s over 2.5s: 2 Hz idle samples only wake the burst, never call speech alone.
const MIN_SAMPLES = 10;
// Scale required transitions with sample count so 8 Hz jitter doesn't trip on noise.
const TRANSITIONS_PER_SAMPLE = 0.3;

/**
 * Aspect-ratio- and yaw-foreshortening-corrected Mouth Aspect Ratio (MAR).
 * Uses inner upper/lower lip centers (13, 14) and mouth corners (61, 291).
 */
export function computeMouthAspectRatio(
  landmarks: NormalizedLandmark[],
  w: number,
  h: number,
  headYawRad = 0
): number {
  const upperLip = landmarks[13];
  const lowerLip = landmarks[14];
  const leftCorner = landmarks[61];
  const rightCorner = landmarks[291];
  if (!upperLip || !lowerLip || !leftCorner || !rightCorner) return 0;

  const cosYaw = Math.max(0.65, Math.cos(headYawRad));
  const mouthWidth = Math.max(
    0.01 * w,
    Math.hypot((rightCorner.x - leftCorner.x) * w, (rightCorner.y - leftCorner.y) * h)
  );
  const lipGap = Math.hypot((lowerLip.x - upperLip.x) * w, (lowerLip.y - upperLip.y) * h);

  // Multiply by cos(yaw) so sideways head turns don't shrink mouthWidth and inflate MAR
  return (lipGap / mouthWidth) * cosYaw;
}

export class MouthMotionTracker {
  private history: Array<{ t: number; mar: number }> = [];

  update(
    landmarks: NormalizedLandmark[],
    w: number,
    h: number,
    t: number,
    headYawRad = 0
  ): MouthMotion {
    const mar = computeMouthAspectRatio(landmarks, w, h, headYawRad);

    if (
      this.history.length > 0 &&
      t - this.history[this.history.length - 1].t > STALE_GAP_MS
    ) {
      this.history.length = 0;
    }

    this.history.push({ t, mar });
    while (this.history.length > 0 && this.history[0].t < t - WINDOW_MS) {
      this.history.shift();
    }

    let transitions = 0;
    let extremum = this.history[0].mar;
    let maxMar = extremum;
    let minMar = extremum;
    let direction: 0 | 1 | -1 = 0; // 1 = opening, -1 = closing

    for (let i = 1; i < this.history.length; i++) {
      const v = this.history[i].mar;
      if (v > maxMar) maxMar = v;
      if (v < minMar) minMar = v;

      if (direction === 0) {
        if (v - extremum >= HYST) {
          direction = 1;
          extremum = v;
          transitions++;
        } else if (extremum - v >= HYST) {
          direction = -1;
          extremum = v;
          transitions++;
        }
      } else if (direction === 1) {
        if (v > extremum) {
          extremum = v;
        } else if (extremum - v >= HYST) {
          direction = -1;
          extremum = v;
          transitions++;
        }
      } else {
        if (v < extremum) {
          extremum = v;
        } else if (v - extremum >= HYST) {
          direction = 1;
          extremum = v;
          transitions++;
        }
      }
    }

    return {
      mar,
      isSpeaking:
        this.history.length >= MIN_SAMPLES &&
        transitions >= Math.ceil(TRANSITIONS_PER_SAMPLE * this.history.length) &&
        maxMar > MIN_OPEN &&
        maxMar - minMar >= MIN_PEAK_TO_PEAK,
      motionScore: transitions,
      isActive: maxMar - minMar >= HYST,
    };
  }

  reset(): void {
    this.history.length = 0;
  }
}
