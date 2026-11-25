import { Vec3 } from './transform-matrix';

export interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
}

export interface IrisGazeOptions {
  /** Video frame aspect ratio (width / height). Defaults to 16 / 9. */
  aspectRatio?: number;
  /** Current head yaw in radians (used to compensate cos(yaw) eye-width foreshortening). */
  headYawRad?: number;
  /**
   * Anatomical/calibrated ratio converting normalized horizontal iris displacement (dx)
   * into sin(theta_h). Nominal physiology: W_canthus (~30 mm) / R_eyeball (~12 mm) = 2.5.
   */
  eyeGainH?: number;
  /**
   * Anatomical/calibrated ratio converting normalized vertical iris displacement (dy)
   * into sin(theta_v). Nominal value ~2.8.
   */
  eyeGainV?: number;
}

export interface IrisGazeResult {
  hRatio: number;      // 0.0 (looking screen-left) to 1.0 (looking screen-right), 0.5 is centered (for UI)
  vRatio: number;      // 0.0 (looking up) to 1.0 (looking down), 0.5 is centered (for UI)
  isBlinking: boolean; // True if both eyes are below blink openness threshold
  rightOpenness: number;
  leftOpenness: number;

  /**
   * Unclamped, aspect-ratio-corrected, yaw-foreshortening-compensated iris displacement
   * normalized by inter-canthal width.
   * - dx > 0: iris shifted toward user's right (screen-right in mirrored webcam view)
   * - dy > 0: iris shifted downward
   */
  dx: number;
  dy: number;

  /** Estimated eye-in-head horizontal rotation angle theta_h (radians, + = screen-right) */
  eyeYawRad: number;
  /** Estimated eye-in-head vertical rotation angle theta_v (radians, + = looking up) */
  eyePitchRad: number;
  /** 3D unit gaze direction vector g_eye in head-local coordinates (+Z points out of face toward screen) */
  eyeVector: Vec3;
  /** Visibility-weighted binocular confidence in [0, 1] */
  confidence: number;
}

/**
 * Key MediaPipe FaceLandmarker (478-landmark) indices for gaze and eyelid calculation:
 * Right eye (user's right, image-left in unmirrored camera frame):
 *   - Outer canthus: 33, Inner canthus: 133, Upper lid: 159, Lower lid: 145, Iris center: 468
 * Left eye (user's left, image-right in unmirrored camera frame):
 *   - Inner canthus: 362, Outer canthus: 263, Upper lid: 386, Lower lid: 374, Iris center: 473
 */
export const LANDMARKS = {
  RIGHT_EYE: {
    OUTER: 33,
    INNER: 133,
    UPPER: 159,
    LOWER: 145,
    IRIS: 468,
  },
  LEFT_EYE: {
    INNER: 362,
    OUTER: 263,
    UPPER: 386,
    LOWER: 374,
    IRIS: 473,
  },
} as const;

const BLINK_OPENNESS_THRESHOLD = 0.15;
/** Default physiological gain W_canthus / R_eyeball (~30 mm / 12 mm = 2.5) */
export const NOMINAL_EYE_GAIN_H = 2.5;
export const NOMINAL_EYE_GAIN_V = 2.8;
const MAX_SIN_EYE_ANGLE = 0.92; // ~67 deg physical clamp to keep arcsin finite

interface SingleEyeMeasurement {
  dx: number;       // Unclamped, aspect-corrected, + = user/screen-right
  dy: number;       // Unclamped, aspect-corrected, + = down
  isOpen: boolean;
  openness: number;
  eyeWidth: number; // Apparent eye width in aspect-corrected units (used for visibility weighting)
}

/**
 * Converts unclamped eye-local displacements (dx, dy) into a 3D spherical eyeball
 * unit vector g_eye in head-local coordinates (+X = camera-right / screen-left, +Y = up, +Z = out of face).
 */
export function irisDisplacementToEyeVector(
  dx: number,
  dy: number,
  eyeGainH: number = NOMINAL_EYE_GAIN_H,
  eyeGainV: number = NOMINAL_EYE_GAIN_V
): { eyeYawRad: number; eyePitchRad: number; eyeVector: Vec3 } {
  // Spherical eyeball kinematics: lateral iris displacement on radius R_e is R_e * sin(theta)
  const sinH = Math.max(-MAX_SIN_EYE_ANGLE, Math.min(MAX_SIN_EYE_ANGLE, dx * eyeGainH));
  const sinV = Math.max(-MAX_SIN_EYE_ANGLE, Math.min(MAX_SIN_EYE_ANGLE, -dy * eyeGainV)); // +pitch is UP

  const eyeYawRad = Math.asin(sinH);
  const eyePitchRad = Math.asin(sinV);

  const cosV = Math.cos(eyePitchRad);
  const cosH = Math.cos(eyeYawRad);

  // In MediaPipe camera/head-local right-handed coordinates:
  //   - Looking screen-right corresponds to moving toward -X_camera (because camera faces the user)
  //   - Looking up corresponds to +Y_camera
  //   - Neutral forward gaze points along +Z_head (out of the face toward the screen)
  const eyeVector: Vec3 = [
    -Math.sin(eyeYawRad) * cosV,
    Math.sin(eyePitchRad),
    cosH * cosV,
  ];

  return { eyeYawRad, eyePitchRad, eyeVector };
}

function calculateSingleEyeGaze(
  landmarks: NormalizedLandmark[],
  leftCornerIdx: number,
  rightCornerIdx: number,
  upperIdx: number,
  lowerIdx: number,
  irisIdx: number,
  invAspectRatio: number,
  cosHeadYaw: number
): SingleEyeMeasurement {
  const leftCorner = landmarks[leftCornerIdx];
  const rightCorner = landmarks[rightCornerIdx];
  const upper = landmarks[upperIdx];
  const lower = landmarks[lowerIdx];
  const iris = landmarks[irisIdx];

  if (!leftCorner || !rightCorner || !upper || !lower || !iris) {
    return { dx: 0, dy: 0, isOpen: false, openness: 0, eyeWidth: 0 };
  }

  // Aspect-ratio correction: scale y by (height / width) so x and y share isotropic pixel units.
  // Without this, roll-invariant basis (u, w) is sheared on 16:9 webcams.
  const lcx = leftCorner.x;
  const lcy = leftCorner.y * invAspectRatio;
  const rcx = rightCorner.x;
  const rcy = rightCorner.y * invAspectRatio;
  const ux0 = upper.x;
  const uy0 = upper.y * invAspectRatio;
  const lx0 = lower.x;
  const ly0 = lower.y * invAspectRatio;
  const ix = iris.x;
  const iy = iris.y * invAspectRatio;

  const ex = rcx - lcx;
  const ey = rcy - lcy;
  const eyeWidth = Math.hypot(ex, ey);
  const eyeHeight = Math.hypot(ux0 - lx0, uy0 - ly0);

  if (eyeWidth < 1e-4) {
    return { dx: 0, dy: 0, isOpen: false, openness: 0, eyeWidth: 0 };
  }

  const openness = eyeHeight / eyeWidth;
  if (openness < BLINK_OPENNESS_THRESHOLD) {
    return { dx: 0, dy: 0, isOpen: false, openness, eyeWidth };
  }

  // Orthonormal roll-invariant eye-local basis in isotropic pixel space
  const u_x = ex / eyeWidth;
  const u_y = ey / eyeWidth;
  const w_x = -u_y;
  const w_y = u_x;

  const centerX = (lcx + rcx) / 2;
  const centerY = (lcy + rcy) / 2;

  const vx = ix - centerX;
  const vy = iy - centerY;

  // Raw projection normalized by apparent 2D eyeWidth
  const rawProjX = (vx * u_x + vy * u_y) / eyeWidth;
  const rawProjY = (vx * w_x + vy * w_y) / eyeWidth;

  // Compensate horizontal foreshortening: when head yaws by psi, projected inter-canthal width
  // shrinks by ~cos(psi), which otherwise inflates rawProjX by 1/cos(psi).
  // Negate rawProjX so dx > 0 means looking screen-right (matching mirrored webcam view).
  const dx = -rawProjX * cosHeadYaw;
  const dy = rawProjY;

  return { dx, dy, isOpen: true, openness, eyeWidth };
}

export function computeIrisGaze(
  landmarks: NormalizedLandmark[],
  previousGaze?: IrisGazeResult,
  options: IrisGazeOptions = {}
): IrisGazeResult {
  const aspectRatio = options.aspectRatio ?? 16 / 9;
  const invAspectRatio = 1 / aspectRatio;
  const headYawRad = options.headYawRad ?? 0;
  // Clamp cos(yaw) floor at 0.65 (~50 deg) to avoid over-damping on extreme noise
  const cosHeadYaw = Math.max(0.65, Math.cos(headYawRad));
  const eyeGainH = options.eyeGainH ?? NOMINAL_EYE_GAIN_H;
  const eyeGainV = options.eyeGainV ?? NOMINAL_EYE_GAIN_V;

  if (!landmarks || landmarks.length < 478) {
    const fallbackDx = previousGaze?.dx ?? 0;
    const fallbackDy = previousGaze?.dy ?? 0;
    const vec = irisDisplacementToEyeVector(fallbackDx, fallbackDy, eyeGainH, eyeGainV);
    return {
      hRatio: previousGaze?.hRatio ?? 0.5,
      vRatio: previousGaze?.vRatio ?? 0.5,
      isBlinking: true,
      rightOpenness: 0,
      leftOpenness: 0,
      dx: fallbackDx,
      dy: fallbackDy,
      ...vec,
      confidence: 0,
    };
  }

  // Right eye: outer (33) is image-left, inner (133) is image-right
  const right = calculateSingleEyeGaze(
    landmarks,
    LANDMARKS.RIGHT_EYE.OUTER,
    LANDMARKS.RIGHT_EYE.INNER,
    LANDMARKS.RIGHT_EYE.UPPER,
    LANDMARKS.RIGHT_EYE.LOWER,
    LANDMARKS.RIGHT_EYE.IRIS,
    invAspectRatio,
    cosHeadYaw
  );

  // Left eye: inner (362) is image-left, outer (263) is image-right
  const left = calculateSingleEyeGaze(
    landmarks,
    LANDMARKS.LEFT_EYE.INNER,
    LANDMARKS.LEFT_EYE.OUTER,
    LANDMARKS.LEFT_EYE.UPPER,
    LANDMARKS.LEFT_EYE.LOWER,
    LANDMARKS.LEFT_EYE.IRIS,
    invAspectRatio,
    cosHeadYaw
  );

  const isBlinking = !right.isOpen && !left.isOpen;

  if (isBlinking && previousGaze) {
    return {
      ...previousGaze,
      isBlinking: true,
      rightOpenness: right.openness,
      leftOpenness: left.openness,
      confidence: 0,
    };
  }

  let dx = 0;
  let dy = 0;
  let confidence = 0;

  if (right.isOpen && left.isOpen) {
    // Visibility-weighted binocular fusion: weight by squared apparent eyeWidth so that
    // under head yaw, the nearer (unoccluded) eye dominates over the foreshortened far eye.
    const wR = right.eyeWidth * right.eyeWidth;
    const wL = left.eyeWidth * left.eyeWidth;
    const wSum = wR + wL;
    dx = (wR * right.dx + wL * left.dx) / wSum;
    dy = (wR * right.dy + wL * left.dy) / wSum;
    confidence = 1.0;
  } else if (right.isOpen) {
    dx = right.dx;
    dy = right.dy;
    confidence = 0.6;
  } else if (left.isOpen) {
    dx = left.dx;
    dy = left.dy;
    confidence = 0.6;
  } else if (previousGaze) {
    dx = previousGaze.dx;
    dy = previousGaze.dy;
    confidence = 0;
  }

  const { eyeYawRad, eyePitchRad, eyeVector } = irisDisplacementToEyeVector(
    dx,
    dy,
    eyeGainH,
    eyeGainV
  );

  // Backwards-compatible [0, 1] UI gauge ratios
  const hRatio = Math.max(0, Math.min(1, 0.5 + dx * 2.5));
  const vRatio = Math.max(0, Math.min(1, 0.5 + dy * 3.0));

  return {
    hRatio,
    vRatio,
    isBlinking,
    rightOpenness: right.openness,
    leftOpenness: left.openness,
    dx,
    dy,
    eyeYawRad,
    eyePitchRad,
    eyeVector,
    confidence,
  };
}