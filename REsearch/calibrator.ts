import {
  EulerAngles,
  Mat3,
  Vec3,
  eulerToRotationMatrix,
  extractRigidTransform,
  mat3MulVec3,
  normalizeVec3,
} from '../math/transform-matrix';
import {
  NOMINAL_EYE_GAIN_H,
  NOMINAL_EYE_GAIN_V,
  irisDisplacementToEyeVector,
} from '../math/iris-gaze';

export type CalibrationPoint =
  | 'center'
  | 'top-left'
  | 'top-center'
  | 'top-right'
  | 'middle-left'
  | 'middle-right'
  | 'bottom-left'
  | 'bottom-center'
  | 'bottom-right'
  | 'vor-left'
  | 'vor-right';

/** Ground-truth normalized screen coordinates [0..1, 0..1] for each calibration target */
export const CALIBRATION_TARGETS: Record<CalibrationPoint, { x: number; y: number }> = {
  center:          { x: 0.50, y: 0.50 },
  'top-left':      { x: 0.08, y: 0.08 },
  'top-center':    { x: 0.50, y: 0.08 },
  'top-right':     { x: 0.92, y: 0.08 },
  'middle-left':   { x: 0.08, y: 0.50 },
  'middle-right':  { x: 0.92, y: 0.50 },
  'bottom-left':   { x: 0.08, y: 0.92 },
  'bottom-center': { x: 0.50, y: 0.92 },
  'bottom-right':  { x: 0.92, y: 0.92 },
  // VOR dynamic calibration steps: eyes remain fixed on screen center while head turns ~15 deg
  'vor-left':      { x: 0.50, y: 0.50 },
  'vor-right':     { x: 0.50, y: 0.50 },
};

export interface CalibrationSample {
  yaw: number;
  pitch: number;
  roll?: number;
  hRatio: number;
  vRatio: number;
  /** Unclamped aspect-corrected iris displacement (preferred over clamped hRatio/vRatio) */
  dx?: number;
  dy?: number;
  /** Optional raw 4x4 column-major MediaPipe facialTransformationMatrix */
  transformMatrix?: number[] | Float32Array;
  point?: CalibrationPoint;
  /** Optional explicit normalized target coordinates (overrides CALIBRATION_TARGETS[point]) */
  targetX?: number;
  targetY?: number;
}

export interface RidgeGazeModel {
  /** 'linear' = [1, X_plane, Y_plane] (3 terms); 'quadratic' = [1, X, Y, X^2, X*Y, Y^2] (6 terms) */
  degree: 1 | 2;
  /** Ridge regression weights mapping phi(X_plane, Y_plane) -> normalized screen x in [0, 1] */
  weightsX: number[];
  /** Ridge regression weights mapping phi(X_plane, Y_plane) -> normalized screen y in [0, 1] */
  weightsY: number[];
  /** Personal horizontal eye-in-head anatomical gain (calibrated via VOR step or nominal 2.5) */
  eyeGainH: number;
  /** Personal vertical eye-in-head anatomical gain (nominal 2.8) */
  eyeGainV: number;
  /** Feature normalization mean/scale for numerical conditioning */
  meanX: number;
  meanY: number;
  scaleX: number;
  scaleY: number;
  /** Regularization strength lambda used during fit */
  lambda: number;
  /** Calibration RMSE in normalized screen units (e.g. 0.04 = 4% of screen width/height) */
  rmseX: number;
  rmseY: number;
}

export interface CalibrationProfile {
  /** Resting neutral head pose (for anatomical guardrail bounds, not tight gaze clipping) */
  neutralYaw: number;
  neutralPitch: number;
  pitchMin: number; // Floor for LOOKING_DOWN
  pitchMax: number;
  yawMin: number;   // Wide anatomical safety bound (not tight screen-edge bound)
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
  /** 3D Ray-Plane + Ridge Regression model (VOR-invariant) */
  ridgeModel?: RidgeGazeModel;
  isAutoBaseline?: boolean;
  calibratedAt: number;
}

export interface CalibrationStep {
  point: CalibrationPoint;
  label: string;
  durationMs: number;
}

/** Legacy 5-point steps (kept for backwards compatibility) */
export const CALIBRATION_STEPS: CalibrationStep[] = [
  { point: 'center', label: 'Look directly at the center of your screen', durationMs: 2000 },
  { point: 'top-left', label: 'Look at the top-left corner of your screen', durationMs: 2000 },
  { point: 'top-right', label: 'Look at the top-right corner of your screen', durationMs: 2000 },
  { point: 'bottom-left', label: 'Look at the bottom-left corner of your screen', durationMs: 2000 },
  { point: 'bottom-right', label: 'Look at the bottom-right corner of your screen', durationMs: 2000 },
];

/**
 * Recommended 9-Point Grid + 2-Step Dynamic VOR Calibration (~17 seconds total).
 * The 9 spatial points solve the 6-parameter 2nd-order screen-plane polynomial;
 * the 2 VOR steps directly identify personal eyeGainH under real head-eye counter-rotation.
 */
export const CALIBRATION_STEPS_9PT_VOR: CalibrationStep[] = [
  { point: 'center',        label: 'Look at the center dot', durationMs: 1500 },
  { point: 'top-left',      label: 'Look at the top-left dot', durationMs: 1400 },
  { point: 'top-center',    label: 'Look at the top-center dot', durationMs: 1400 },
  { point: 'top-right',     label: 'Look at the top-right dot', durationMs: 1400 },
  { point: 'middle-right',  label: 'Look at the middle-right dot', durationMs: 1400 },
  { point: 'bottom-right',  label: 'Look at the bottom-right dot', durationMs: 1400 },
  { point: 'bottom-center', label: 'Look at the bottom-center dot', durationMs: 1400 },
  { point: 'bottom-left',   label: 'Look at the bottom-left dot', durationMs: 1400 },
  { point: 'middle-left',   label: 'Look at the middle-left dot', durationMs: 1400 },
  { point: 'vor-left',      label: 'Keep eyes on CENTER while turning head slightly LEFT', durationMs: 2200 },
  { point: 'vor-right',     label: 'Keep eyes on CENTER while turning head slightly RIGHT', durationMs: 2200 },
];

export interface WorldGazeRay {
  /** 3D unit gaze vector g_world = R_head * g_eye in camera/world coordinates */
  gazeWorld: Vec3;
  /** Metric screen-plane intersection (X_plane, Y_plane) on z = 0 in cm (+X = screen-right, +Y = screen-down) */
  planeX: number;
  planeY: number;
  /** Equivalent world gaze angles in degrees (for diagnostics / fallback) */
  gxDeg: number;
  gyDeg: number;
}

const DEFAULT_VIEWING_DISTANCE_CM = 55;
const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;
const MIN_GZ = 0.15; // Guard against near-parallel rays to screen plane

/**
 * Composes 3D head pose (R_head, t_head) with spherical eye-in-head vector g_eye
 * via g_world = R_head * g_eye, then intersects the 3D ray with the display plane z = 0:
 *
 *   P_screen = o_world + s * g_world,   s = -o_z / g_world_z
 *
 * Returns screen-aligned plane coordinates (+planeX = screen-right, +planeY = screen-down).
 */
export function composeWorldGazeRay(
  sample: {
    yaw: number;
    pitch: number;
    roll?: number;
    hRatio: number;
    vRatio: number;
    dx?: number;
    dy?: number;
    transformMatrix?: number[] | Float32Array;
  },
  eyeGainH: number = NOMINAL_EYE_GAIN_H,
  eyeGainV: number = NOMINAL_EYE_GAIN_V
): WorldGazeRay {
  // Prefer unclamped dx/dy; fall back to recovering from hRatio/vRatio if legacy caller
  const dx = sample.dx !== undefined ? sample.dx : (sample.hRatio - 0.5) / 2.5;
  const dy = sample.dy !== undefined ? sample.dy : (sample.vRatio - 0.5) / 3.0;

  let R_head: Mat3;
  let t_head: Vec3;

  if (sample.transformMatrix && sample.transformMatrix.length >= 16) {
    const tf = extractRigidTransform(sample.transformMatrix);
    R_head = tf.rotation;
    t_head = tf.translation;
  } else {
    R_head = eulerToRotationMatrix({
      yaw: sample.yaw,
      pitch: sample.pitch,
      roll: sample.roll ?? 0,
    });
    t_head = [0, 0, -DEFAULT_VIEWING_DISTANCE_CM];
  }

  const { eyeVector } = irisDisplacementToEyeVector(dx, dy, eyeGainH, eyeGainV);
  const gWorld = normalizeVec3(mat3MulVec3(R_head, eyeVector));

  // Clamp gWorld[2] so rays nearly parallel to the screen don't blow up to infinity
  const gz = Math.max(MIN_GZ, gWorld[2]);
  const s = -t_head[2] / gz; // Positive distance along ray to z = 0 plane (t_head[2] < 0)

  // Intersect ray with z = 0 in camera coordinates, then flip to screen-aligned axes:
  //   screen-right = -X_camera, screen-down = -Y_camera
  const hitCamX = t_head[0] + s * gWorld[0];
  const hitCamY = t_head[1] + s * gWorld[1];
  const planeX = -hitCamX;
  const planeY = -hitCamY;

  const gxDeg = Math.atan2(-gWorld[0], gz) * RAD_TO_DEG;
  const gyDeg = Math.atan2(-gWorld[1], gz) * RAD_TO_DEG;

  return { gazeWorld: gWorld, planeX, planeY, gxDeg, gyDeg };
}

/**
 * Backwards-compatible combined gaze helper, now powered by true SO(3) 3D vector
 * composition (g_world = R_head * g_eye) instead of the fixed linear gain (-yaw + (h-0.5)*40).
 */
export function computeCombinedGaze(
  yaw: number,
  pitch: number,
  hRatio: number,
  vRatio: number,
  options?: {
    roll?: number;
    dx?: number;
    dy?: number;
    transformMatrix?: number[] | Float32Array;
    eyeGainH?: number;
    eyeGainV?: number;
  }
): { gx: number; gy: number; planeX: number; planeY: number } {
  const ray = composeWorldGazeRay(
    {
      yaw,
      pitch,
      roll: options?.roll,
      hRatio,
      vRatio,
      dx: options?.dx,
      dy: options?.dy,
      transformMatrix: options?.transformMatrix,
    },
    options?.eyeGainH ?? NOMINAL_EYE_GAIN_H,
    options?.eyeGainV ?? NOMINAL_EYE_GAIN_V
  );
  return { gx: ray.gxDeg, gy: ray.gyDeg, planeX: ray.planeX, planeY: ray.planeY };
}

/**
 * Builds the polynomial design vector phi(x, y) on standardized screen-plane coordinates.
 *   degree = 1 -> [1, nx, ny]
 *   degree = 2 -> [1, nx, ny, nx^2, nx*ny, ny^2]
 */
function buildFeatureVector(nx: number, ny: number, degree: 1 | 2): number[] {
  if (degree === 1) {
    return [1, nx, ny];
  }
  return [1, nx, ny, nx * nx, nx * ny, ny * ny];
}

/**
 * Solves (X^T X + lambda * I_reg) w = X^T y via Gaussian elimination with partial pivoting.
 * Intercept weight w[0] is NOT penalized (I_reg[0][0] = 0).
 * Runs in <0.1 ms for p <= 6 in a browser Web Worker.
 */
function solveRidgeRegression(X: number[][], y: number[], lambda: number): number[] {
  const n = X.length;
  const p = X[0].length;

  // Assemble normal equations A = X^T X + lambda * I_reg, b = X^T y
  const A: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const b: number[] = new Array(p).fill(0);

  for (let i = 0; i < n; i++) {
    const row = X[i];
    const yi = y[i];
    for (let j = 0; j < p; j++) {
      b[j] += row[j] * yi;
      for (let k = j; k < p; k++) {
        A[j][k] += row[j] * row[k];
      }
    }
  }

  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) {
      A[j][k] = A[k][j];
    }
    if (j > 0) {
      A[j][j] += lambda; // Do not regularize bias term w[0]
    }
  }

  // Augmented matrix Gaussian elimination with partial pivoting
  const M = A.map((row, idx) => [...row, b[idx]]);
  for (let col = 0; col < p; col++) {
    let pivotRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let r = col + 1; r < p; r++) {
      const val = Math.abs(M[r][col]);
      if (val > maxVal) {
        maxVal = val;
        pivotRow = r;
      }
    }
    if (pivotRow !== col) {
      const tmp = M[col];
      M[col] = M[pivotRow];
      M[pivotRow] = tmp;
    }

    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-10) {
      continue;
    }

    for (let r = col + 1; r < p; r++) {
      const factor = M[r][col] / pivot;
      for (let c = col; c <= p; c++) {
        M[r][c] -= factor * M[col][c];
      }
    }
  }

  // Back-substitution
  const w = new Array(p).fill(0);
  for (let i = p - 1; i >= 0; i--) {
    let sum = M[i][p];
    for (let j = i + 1; j < p; j++) {
      sum -= M[i][j] * w[j];
    }
    w[i] = Math.abs(M[i][i]) > 1e-10 ? sum / M[i][i] : 0;
  }

  return w;
}

/**
 * Estimates the candidate's personal anatomical eye-in-head gain k_h from VOR calibration
 * samples ('vor-left', 'vor-right', 'center'). During VOR fixation on screen center:
 *   sin(theta_h(t) - theta_h0) = -sin(yaw(t) - yaw0)  (exact counter-rotation in SO(3))
 * Since sin(theta_h) = k_h * dx, we solve the 1D ridge-regularized least-squares fit for k_h.
 */
export function fitVorEyeGain(samples: CalibrationSample[]): number {
  const vorSamples = samples.filter(
    (s) => s.point === 'vor-left' || s.point === 'vor-right' || s.point === 'center'
  );
  if (vorSamples.length < 3) return NOMINAL_EYE_GAIN_H;

  const centerSamples = samples.filter((s) => s.point === 'center');
  const refYaw =
    centerSamples.length > 0
      ? percentile(centerSamples.map((s) => s.yaw), 50)
      : percentile(vorSamples.map((s) => s.yaw), 50);

  const getDx = (s: CalibrationSample) =>
    s.dx !== undefined ? s.dx : (s.hRatio - 0.5) / 2.5;
  const refDx =
    centerSamples.length > 0
      ? percentile(centerSamples.map(getDx), 50)
      : percentile(vorSamples.map(getDx), 50);

  // Require sufficient head-yaw excitation (>= 6 deg spread) to avoid dividing by noise
  const yawSpread =
    percentile(vorSamples.map((s) => s.yaw), 90) -
    percentile(vorSamples.map((s) => s.yaw), 10);
  if (yawSpread < 6) return NOMINAL_EYE_GAIN_H;

  // When head turns toward screen-right (dYaw > 0), eyes counter-rotate screen-left (dDx < 0),
  // so target sin(delta_theta_h) = -sin(dYawRad). Shrink lightly toward NOMINAL_EYE_GAIN_H.
  let num = 0;
  let den = 0;
  const priorWeight = 0.001;

  for (const s of vorSamples) {
    const dYawRad = (s.yaw - refYaw) * DEG_TO_RAD;
    const dDx = getDx(s) - refDx;
    const targetSin = -Math.sin(dYawRad);
    num += dDx * targetSin;
    den += dDx * dDx;
  }

  const kHat = (num + priorWeight * NOMINAL_EYE_GAIN_H) / (den + priorWeight);
  // Clamp to physiological bounds [1.5, 4.2] (roughly W_canthus / R_e across human anatomy)
  return Math.max(1.5, Math.min(4.2, kHat));
}

/**
 * Slowly absorbs minor posture shifts (slouching, leaning) in normalized screen coordinates
 * while clamping drift to safe physical limits so deliberate off-screen glances cannot drift the baseline.
 */
export class AdaptiveDriftTracker {
  private driftX = 0;     // Normalized screen units (max +-0.08 = 8% of screen width)
  private driftY = 0;     // Normalized screen units (max +-0.06 = 6% of screen height)
  private driftPitch = 0; // Degrees (for LOOKING_DOWN pitch floor)

  private readonly alpha = 0.012; // ~16-20s adaptation time constant at 5 Hz
  private readonly maxDriftScreenX = 0.08;
  private readonly maxDriftScreenY = 0.06;
  private readonly maxDriftPitch = 6;

  reset(): void {
    this.driftX = 0;
    this.driftY = 0;
    this.driftPitch = 0;
  }

  update(
    rawScreenX: number,
    rawScreenY: number,
    pitch: number,
    profile: CalibrationProfile,
    isStableInBounds: boolean
  ): void {
    if (!isStableInBounds) return;

    // Only adapt when gaze is comfortably near the interior of the screen [0.15, 0.85]
    if (rawScreenX < 0.15 || rawScreenX > 0.85 || rawScreenY < 0.15 || rawScreenY > 0.85) {
      return;
    }

    const targetX = Math.max(-this.maxDriftScreenX, Math.min(this.maxDriftScreenX, rawScreenX - 0.5));
    const targetY = Math.max(-this.maxDriftScreenY, Math.min(this.maxDriftScreenY, rawScreenY - 0.5));
    const targetPitch = Math.max(
      -this.maxDriftPitch,
      Math.min(this.maxDriftPitch, pitch - profile.neutralPitch)
    );

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
 * Maps a live pose + iris observation to normalized screen coordinates [0..1, 0..1]
 * using the 3D ray-plane + Ridge Regression model when calibrated, or the fallback
 * 3D angular screen bounds for auto-baseline profiles.
 */
export function mapGazeToScreen(
  input:
    | {
        yaw: number;
        pitch: number;
        roll?: number;
        hRatio: number;
        vRatio: number;
        dx?: number;
        dy?: number;
        transformMatrix?: number[] | Float32Array;
      }
    | number,
  gyOrProfile: number | CalibrationProfile,
  profileOrDrift?: CalibrationProfile | { driftX: number; driftY: number },
  maybeDrift?: { driftX: number; driftY: number }
): { x: number; y: number; rawX: number; rawY: number; isOffScreen: boolean; planeX?: number; planeY?: number } {
  const MARGIN = 0.10;

  // Overload 1 (recommended): mapGazeToScreen(sampleObj, profile, drift?)
  if (typeof input === 'object') {
    const profile = gyOrProfile as CalibrationProfile;
    const drift = (profileOrDrift as { driftX: number; driftY: number }) ?? { driftX: 0, driftY: 0 };

    const eyeGainH = profile.ridgeModel?.eyeGainH ?? NOMINAL_EYE_GAIN_H;
    const eyeGainV = profile.ridgeModel?.eyeGainV ?? NOMINAL_EYE_GAIN_V;
    const ray = composeWorldGazeRay(input, eyeGainH, eyeGainV);

    if (profile.ridgeModel) {
      const m = profile.ridgeModel;
      const nx = (ray.planeX - m.meanX) / m.scaleX;
      const ny = (ray.planeY - m.meanY) / m.scaleY;
      const phi = buildFeatureVector(nx, ny, m.degree);

      let rawX = 0;
      let rawY = 0;
      for (let i = 0; i < phi.length; i++) {
        rawX += m.weightsX[i] * phi[i];
        rawY += m.weightsY[i] * phi[i];
      }

      const x = rawX - drift.driftX;
      const y = rawY - drift.driftY;
      const isOffScreen = x < -MARGIN || x > 1 + MARGIN || y < -MARGIN || y > 1 + MARGIN;
      return { x, y, rawX, rawY, isOffScreen, planeX: ray.planeX, planeY: ray.planeY };
    }

    // Fallback for auto-baseline (before explicit calibration)
    const { gxMin, gxMax, gyMin, gyMax } = profile.screenBounds;
    const spanX = Math.max(10, gxMax - gxMin);
    const spanY = Math.max(8, gyMax - gyMin);
    const rawX = (ray.gxDeg - gxMin) / spanX;
    const rawY = (ray.gyDeg - gyMin) / spanY;
    const x = rawX - drift.driftX;
    const y = rawY - drift.driftY;
    const isOffScreen = x < -MARGIN || x > 1 + MARGIN || y < -MARGIN || y > 1 + MARGIN;
    return { x, y, rawX, rawY, isOffScreen, planeX: ray.planeX, planeY: ray.planeY };
  }

  // Overload 2 (legacy): mapGazeToScreen(gx, gy, profile, drift?)
  const gx = input;
  const gy = gyOrProfile as number;
  const profile = profileOrDrift as CalibrationProfile;
  const drift = maybeDrift ?? { driftX: 0, driftY: 0 };

  const { gxMin, gxMax, gyMin, gyMax } = profile.screenBounds;
  const spanX = Math.max(10, gxMax - gxMin);
  const spanY = Math.max(8, gyMax - gyMin);
  const rawX = (gx - gxMin) / spanX;
  const rawY = (gy - gyMin) / spanY;
  const x = rawX - drift.driftX;
  const y = rawY - drift.driftY;
  const isOffScreen = x < -MARGIN || x > 1 + MARGIN || y < -MARGIN || y > 1 + MARGIN;
  return { x, y, rawX, rawY, isOffScreen };
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
 * Drops the first 30% of samples within each calibration step to remove saccade-settling
 * transients before regression fitting.
 */
function trimSaccadeSettling(samples: CalibrationSample[]): CalibrationSample[] {
  const byPoint = new Map<string, CalibrationSample[]>();
  for (const s of samples) {
    const key = s.point ?? 'unlabeled';
    const arr = byPoint.get(key) ?? [];
    arr.push(s);
    byPoint.set(key, arr);
  }

  const trimmed: CalibrationSample[] = [];
  for (const group of byPoint.values()) {
    const dropCount = group.length >= 4 ? Math.floor(group.length * 0.3) : 0;
    trimmed.push(...group.slice(dropCount));
  }
  return trimmed;
}

/**
 * Builds an automatic initial baseline from the candidate's resting posture
 * during the first ~2 seconds before explicit calibration is run.
 */
export function buildAutoBaselineProfile(samples: CalibrationSample[]): CalibrationProfile {
  const medYaw = percentile(samples.map((s) => s.yaw), 50);
  const medPitch = percentile(samples.map((s) => s.pitch), 50);
  const medH = percentile(samples.map((s) => s.hRatio), 50);
  const medV = percentile(samples.map((s) => s.vRatio), 50);
  const { gx, gy } = computeCombinedGaze(medYaw, medPitch, medH, medV);

  return {
    neutralYaw: medYaw,
    neutralPitch: medPitch,
    // Wide anatomical safety guardrails (±35 deg yaw, +28 deg pitch up) so VOR head turns
    // do not falsely trigger HEAD_OFF_SCREEN; pitchMin stays tighter for LOOKING_DOWN.
    yawMin: medYaw - 35,
    yawMax: medYaw + 35,
    pitchMin: medPitch - 15,
    pitchMax: medPitch + 28,
    hRatioMin: Math.max(0.05, medH - 0.25),
    hRatioMax: Math.min(0.95, medH + 0.25),
    vRatioMin: Math.max(0.05, medV - 0.25),
    vRatioMax: Math.min(0.95, medV + 0.25),
    screenBounds: {
      gxMin: gx - 16,
      gxMax: gx + 16,
      gyMin: gy - 12,
      gyMax: gy + 12,
    },
    isAutoBaseline: true,
    calibratedAt: Date.now(),
  };
}

/**
 * Fits the VOR-compensated 3D Ray-Plane + Ridge Regression gaze calibration model.
 * - Supports both legacy 5-point calibration (degree-1 affine plane fit, 3 params/axis)
 *   and recommended 9-point + 2-VOR calibration (degree-2 polynomial fit, 6 params/axis
 *   + personalized eyeGainH).
 */
export function buildCalibrationProfile(samples: CalibrationSample[]): CalibrationProfile {
  if (samples.length === 0) {
    return buildAutoBaselineProfile([{ yaw: 0, pitch: 0, hRatio: 0.5, vRatio: 0.5 }]);
  }

  const cleanSamples = trimSaccadeSettling(samples);
  const eyeGainH = fitVorEyeGain(cleanSamples);
  const eyeGainV = NOMINAL_EYE_GAIN_V;

  // Resting neutral pose (from 'center' step if available, else median across all samples)
  const centerSamples = cleanSamples.filter((s) => s.point === 'center');
  const neutralYaw =
    centerSamples.length > 0
      ? percentile(centerSamples.map((s) => s.yaw), 50)
      : percentile(cleanSamples.map((s) => s.yaw), 50);
  const neutralPitch =
    centerSamples.length > 0
      ? percentile(centerSamples.map((s) => s.pitch), 50)
      : percentile(cleanSamples.map((s) => s.pitch), 50);

  // Project all clean samples onto the z = 0 screen plane via 3D ray intersection
  const projected = cleanSamples.map((s) => {
    const ray = composeWorldGazeRay(s, eyeGainH, eyeGainV);
    const target =
      s.targetX !== undefined && s.targetY !== undefined
        ? { x: s.targetX, y: s.targetY }
        : s.point
          ? CALIBRATION_TARGETS[s.point]
          : undefined;
    return { sample: s, ray, target };
  });

  const labeled = projected.filter((p) => p.target !== undefined);
  const distinctSpatialPoints = new Set(
    labeled
      .filter((p) => p.sample.point !== 'vor-left' && p.sample.point !== 'vor-right')
      .map((p) => `${p.target!.x.toFixed(2)},${p.target!.y.toFixed(2)}`)
  ).size;

  let ridgeModel: RidgeGazeModel | undefined;

  if (labeled.length >= 5 && distinctSpatialPoints >= 4) {
    // Use 2nd-order polynomial (6 params) only when >= 7 distinct spatial targets exist (e.g. 9-point);
    // otherwise use 1st-order affine mapping (3 params) to avoid overfitting on 5-point calibration.
    const degree: 1 | 2 = distinctSpatialPoints >= 7 ? 2 : 1;
    const lambda = degree === 2 ? 0.05 : 0.02;

    const pxArr = labeled.map((p) => p.ray.planeX);
    const pyArr = labeled.map((p) => p.ray.planeY);
    const meanX = pxArr.reduce((a, b) => a + b, 0) / pxArr.length;
    const meanY = pyArr.reduce((a, b) => a + b, 0) / pyArr.length;
    const stdX = Math.sqrt(pxArr.reduce((a, x) => a + (x - meanX) ** 2, 0) / pxArr.length);
    const stdY = Math.sqrt(pyArr.reduce((a, y) => a + (y - meanY) ** 2, 0) / pyArr.length);
    const scaleX = Math.max(2.0, stdX);
    const scaleY = Math.max(2.0, stdY);

    const X = labeled.map((p) =>
      buildFeatureVector((p.ray.planeX - meanX) / scaleX, (p.ray.planeY - meanY) / scaleY, degree)
    );
    const yX = labeled.map((p) => p.target!.x);
    const yY = labeled.map((p) => p.target!.y);

    const weightsX = solveRidgeRegression(X, yX, lambda);
    const weightsY = solveRidgeRegression(X, yY, lambda);

    // Compute in-sample RMSE for quality telemetry
    let sqErrX = 0;
    let sqErrY = 0;
    for (let i = 0; i < X.length; i++) {
      let predX = 0;
      let predY = 0;
      for (let j = 0; j < X[i].length; j++) {
        predX += weightsX[j] * X[i][j];
        predY += weightsY[j] * X[i][j];
      }
      sqErrX += (predX - yX[i]) ** 2;
      sqErrY += (predY - yY[i]) ** 2;
    }

    ridgeModel = {
      degree,
      weightsX,
      weightsY,
      eyeGainH,
      eyeGainV,
      meanX,
      meanY,
      scaleX,
      scaleY,
      lambda,
      rmseX: Math.sqrt(sqErrX / X.length),
      rmseY: Math.sqrt(sqErrY / X.length),
    };
  }

  // Fallback screenBounds (in 3D world-gaze degrees) & wide anatomical head guardrails
  const pitches = cleanSamples.map((s) => s.pitch);
  const hRatios = cleanSamples.map((s) => s.hRatio);
  const vRatios = cleanSamples.map((s) => s.vRatio);

  const pitchP15 = percentile(pitches, 15);
  const gxVals = projected.map((p) => p.ray.gxDeg);
  const gyVals = projected.map((p) => p.ray.gyDeg);
  const gxMin = percentile(gxVals, 10);
  const gxMax = percentile(gxVals, 90);
  const gyMin = percentile(gyVals, 10);
  const gyMax = percentile(gyVals, 90);

  return {
    neutralYaw,
    neutralPitch,
    // Wide anatomical yaw/pitch-up ceilings so natural VOR head turns are handled by ridgeModel
    yawMin: neutralYaw - 35,
    yawMax: neutralYaw + 35,
    pitchMin: Math.min(pitchP15 - 6, neutralPitch - 14),
    pitchMax: neutralPitch + 28,
    hRatioMin: Math.max(0.05, percentile(hRatios, 15) - 0.08),
    hRatioMax: Math.min(0.95, percentile(hRatios, 85) + 0.08),
    vRatioMin: Math.max(0.05, percentile(vRatios, 15) - 0.08),
    vRatioMax: Math.min(0.95, percentile(vRatios, 85) + 0.08),
    screenBounds: {
      gxMin: Math.min(gxMin, gxMax - 10),
      gxMax: Math.max(gxMax, gxMin + 10),
      gyMin: Math.min(gyMin, gyMax - 8),
      gyMax: Math.max(gyMax, gyMin + 8),
    },
    ridgeModel,
    isAutoBaseline: false,
    calibratedAt: Date.now(),
  };
}