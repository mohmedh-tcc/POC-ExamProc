export type CalibrationPoint = 'center' | 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';

export interface CalibrationSample {
  yaw: number;
  pitch: number;
  hRatio: number;
  vRatio: number;
  point?: CalibrationPoint;
}

export interface CalibrationProfile {
  pitchMin: number; // Floor for LOOKING_DOWN
  pitchMax: number;
  yawMin: number;
  yawMax: number;
  hRatioMin: number;
  hRatioMax: number;
  vRatioMin: number;
  vRatioMax: number;
  screenBounds: {
    gxMin: number;
    gxMax: number;
    gyMin: number;
    gyMax: number;
  };
  isAutoBaseline?: boolean;
  calibratedAt: number;
}

export interface CalibrationStep {
  point: CalibrationPoint;
  label: string;
  durationMs: number;
}

export const CALIBRATION_STEPS: CalibrationStep[] = [
  { point: 'center', label: 'Look directly at the center of your screen', durationMs: 2000 },
  { point: 'top-left', label: 'Look at the top-left corner of your screen', durationMs: 2000 },
  { point: 'top-right', label: 'Look at the top-right corner of your screen', durationMs: 2000 },
  { point: 'bottom-left', label: 'Look at the bottom-left corner of your screen', durationMs: 2000 },
  { point: 'bottom-right', label: 'Look at the bottom-right corner of your screen', durationMs: 2000 },
];

/**
 * Combines head pose (yaw/pitch) and iris displacement (hRatio/vRatio) into a single
 * degree-equivalent gaze vector.
 * - gx increases when looking right (either head turn or iris shift)
 * - gy increases when looking down (either head tilt or iris shift)
 */
export function computeCombinedGaze(
  yaw: number,
  pitch: number,
  hRatio: number,
  vRatio: number
): { gx: number; gy: number } {
  const gx = -yaw + (hRatio - 0.5) * 40;
  const gy = -pitch + (vRatio - 0.5) * 35;
  return { gx, gy };
}

/**
 * Slowly absorbs minor posture shifts (slouching, leaning) while clamping drift
 * to safe physical limits so deliberate off-screen glances cannot drift the baseline.
 */
export class AdaptiveDriftTracker {
  private driftX = 0;
  private driftY = 0;
  private driftPitch = 0;

  private readonly alpha = 0.015; // ~15-20s adaptation at 5 Hz
  private readonly maxDriftX = 8; // degrees
  private readonly maxDriftY = 6; // degrees

  reset(): void {
    this.driftX = 0;
    this.driftY = 0;
    this.driftPitch = 0;
  }

  update(
    gx: number,
    gy: number,
    pitch: number,
    profile: CalibrationProfile,
    isStableInBounds: boolean
  ): void {
    if (!isStableInBounds) return;

    const centerX = (profile.screenBounds.gxMin + profile.screenBounds.gxMax) / 2;
    const centerY = (profile.screenBounds.gyMin + profile.screenBounds.gyMax) / 2;
    const centerPitch = (profile.pitchMin + profile.pitchMax) / 2;

    const targetX = Math.max(-this.maxDriftX, Math.min(this.maxDriftX, gx - centerX));
    const targetY = Math.max(-this.maxDriftY, Math.min(this.maxDriftY, gy - centerY));
    const targetPitch = Math.max(-this.maxDriftY, Math.min(this.maxDriftY, pitch - centerPitch));

    this.driftX += this.alpha * (targetX - this.driftX);
    this.driftY += this.alpha * (targetY - this.driftY);
    this.driftPitch += this.alpha * (targetPitch - this.driftPitch);
  }

  getDrift(): { driftX: number; driftY: number; driftPitch: number } {
    return {
      driftX: this.driftX,
      driftY: this.driftY,
      driftPitch: this.driftPitch,
    };
  }
}

/**
 * Maps a combined gaze vector (gx, gy) to normalized screen coordinates [0..1, 0..1],
 * compensated by the adaptive posture drift offset.
 */
export function mapGazeToScreen(
  gx: number,
  gy: number,
  profile: CalibrationProfile,
  drift: { driftX: number; driftY: number } = { driftX: 0, driftY: 0 }
): { x: number; y: number; isOffScreen: boolean } {
  const { gxMin, gxMax, gyMin, gyMax } = profile.screenBounds;
  const spanX = Math.max(8, gxMax - gxMin);
  const spanY = Math.max(6, gyMax - gyMin);

  const adjGx = gx - drift.driftX;
  const adjGy = gy - drift.driftY;

  const x = (adjGx - gxMin) / spanX;
  const y = (adjGy - gyMin) / spanY;

  const MARGIN = 0.08;
  const isOffScreen = x < -MARGIN || x > 1 + MARGIN || y < -MARGIN || y > 1 + MARGIN;

  return { x, y, isOffScreen };
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

/**
 * Builds an automatic initial baseline from the candidate's resting posture
 * during the first ~2 seconds before explicit 5-point calibration is run.
 */
export function buildAutoBaselineProfile(samples: CalibrationSample[]): CalibrationProfile {
  const medYaw = percentile(samples.map((s) => s.yaw), 50);
  const medPitch = percentile(samples.map((s) => s.pitch), 50);
  const medH = percentile(samples.map((s) => s.hRatio), 50);
  const medV = percentile(samples.map((s) => s.vRatio), 50);
  const { gx, gy } = computeCombinedGaze(medYaw, medPitch, medH, medV);

  return {
    yawMin: medYaw - 18,
    yawMax: medYaw + 18,
    pitchMin: medPitch - 14,
    pitchMax: medPitch + 16,
    hRatioMin: Math.max(0.05, medH - 0.22),
    hRatioMax: Math.min(0.95, medH + 0.22),
    vRatioMin: Math.max(0.05, medV - 0.22),
    vRatioMax: Math.min(0.95, medV + 0.22),
    screenBounds: {
      gxMin: gx - 14,
      gxMax: gx + 14,
      gyMin: gy - 11,
      gyMax: gy + 11,
    },
    isAutoBaseline: true,
    calibratedAt: Date.now(),
  };
}

/**
 * Computes a personal bounding envelope using 15th-85th percentiles and tight margins,
 * plus a 2D screen-plane combined gaze mapping.
 */
export function buildCalibrationProfile(samples: CalibrationSample[]): CalibrationProfile {
  if (samples.length === 0) {
    return buildAutoBaselineProfile([{ yaw: 0, pitch: 0, hRatio: 0.5, vRatio: 0.5 }]);
  }

  const yaws = samples.map((s) => s.yaw);
  const pitches = samples.map((s) => s.pitch);
  const hRatios = samples.map((s) => s.hRatio);
  const vRatios = samples.map((s) => s.vRatio);

  const yawP15 = percentile(yaws, 15);
  const yawP85 = percentile(yaws, 85);

  const pitchP15 = percentile(pitches, 15);
  const pitchP85 = percentile(pitches, 85);

  const hRatioP15 = percentile(hRatios, 15);
  const hRatioP85 = percentile(hRatios, 85);

  const vRatioP15 = percentile(vRatios, 15);
  const vRatioP85 = percentile(vRatios, 85);

  const YAW_MARGIN = 6;
  const PITCH_MARGIN = 5;
  const IRIS_MARGIN = 0.08;

  const combined = samples.map((s) => ({
    ...computeCombinedGaze(s.yaw, s.pitch, s.hRatio, s.vRatio),
    point: s.point,
  }));

  const leftSamples = combined.filter((c) => c.point === 'top-left' || c.point === 'bottom-left');
  const rightSamples = combined.filter((c) => c.point === 'top-right' || c.point === 'bottom-right');
  const topSamples = combined.filter((c) => c.point === 'top-left' || c.point === 'top-right');
  const bottomSamples = combined.filter((c) => c.point === 'bottom-left' || c.point === 'bottom-right');

  const gxMin = leftSamples.length > 0
    ? percentile(leftSamples.map((c) => c.gx), 50)
    : percentile(combined.map((c) => c.gx), 15);
  const gxMax = rightSamples.length > 0
    ? percentile(rightSamples.map((c) => c.gx), 50)
    : percentile(combined.map((c) => c.gx), 85);
  const gyMin = topSamples.length > 0
    ? percentile(topSamples.map((c) => c.gy), 50)
    : percentile(combined.map((c) => c.gy), 15);
  const gyMax = bottomSamples.length > 0
    ? percentile(bottomSamples.map((c) => c.gy), 50)
    : percentile(combined.map((c) => c.gy), 85);

  return {
    yawMin: yawP15 - YAW_MARGIN,
    yawMax: yawP85 + YAW_MARGIN,
    pitchMin: pitchP15 - PITCH_MARGIN,
    pitchMax: pitchP85 + PITCH_MARGIN,
    hRatioMin: Math.max(0.05, hRatioP15 - IRIS_MARGIN),
    hRatioMax: Math.min(0.95, hRatioP85 + IRIS_MARGIN),
    vRatioMin: Math.max(0.05, vRatioP15 - IRIS_MARGIN),
    vRatioMax: Math.min(0.95, vRatioP85 + IRIS_MARGIN),
    screenBounds: {
      gxMin: Math.min(gxMin, gxMax - 8),
      gxMax: Math.max(gxMax, gxMin + 8),
      gyMin: Math.min(gyMin, gyMax - 6),
      gyMax: Math.max(gyMax, gyMin + 6),
    },
    isAutoBaseline: false,
    calibratedAt: Date.now(),
  };
}
