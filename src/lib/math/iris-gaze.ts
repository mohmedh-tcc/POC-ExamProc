export interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
}

export interface IrisGazeResult {
  hRatio: number;      // 0.0 (looking left) to 1.0 (looking right), 0.5 is centered
  vRatio: number;      // 0.0 (looking up) to 1.0 (looking down), 0.5 is centered
  isBlinking: boolean; // True if eye openness ratio < threshold
  rightOpenness: number;
  leftOpenness: number;
}

/**
 * Key landmark indices for gaze and eyelid calculation:
 * Right eye:
 *   - Outer canthus: 33
 *   - Inner canthus: 133
 *   - Upper eyelid: 159
 *   - Lower eyelid: 145
 *   - Iris center: 468
 * 
 * Left eye:
 *   - Inner canthus: 362
 *   - Outer canthus: 263
 *   - Upper eyelid: 386
 *   - Lower eyelid: 374
 *   - Iris center: 473
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

const BLINK_OPENNESS_THRESHOLD = 0.15; // Eye height / width below this ratio signifies blink/closed eye

function calculateSingleEyeGaze(
  landmarks: NormalizedLandmark[],
  leftCornerIdx: number,
  rightCornerIdx: number,
  upperIdx: number,
  lowerIdx: number,
  irisIdx: number,
  previousResult?: { h: number; v: number }
): { hRatio: number; vRatio: number; isOpen: boolean; openness: number } {
  const leftCorner = landmarks[leftCornerIdx];
  const rightCorner = landmarks[rightCornerIdx];
  const upper = landmarks[upperIdx];
  const lower = landmarks[lowerIdx];
  const iris = landmarks[irisIdx];

  if (!leftCorner || !rightCorner || !upper || !lower || !iris) {
    return {
      hRatio: previousResult?.h ?? 0.5,
      vRatio: previousResult?.v ?? 0.5,
      isOpen: false,
      openness: 0,
    };
  }

  // Eye axis vector from image-left corner to image-right corner
  const ex = rightCorner.x - leftCorner.x;
  const ey = rightCorner.y - leftCorner.y;
  const eyeWidth = Math.hypot(ex, ey);
  const eyeHeight = Math.hypot(upper.x - lower.x, upper.y - lower.y);

  if (eyeWidth < 1e-4) {
    return {
      hRatio: previousResult?.h ?? 0.5,
      vRatio: previousResult?.v ?? 0.5,
      isOpen: false,
      openness: 0,
    };
  }

  const openness = eyeHeight / eyeWidth;
  if (openness < BLINK_OPENNESS_THRESHOLD) {
    return {
      hRatio: previousResult?.h ?? 0.5,
      vRatio: previousResult?.v ?? 0.5,
      isOpen: false,
      openness,
    };
  }

  // Orthonormal eye-local basis (u = along eye axis, w = perpendicular downward)
  // This makes gaze ratios invariant to head tilt (roll).
  const ux = ex / eyeWidth;
  const uy = ey / eyeWidth;
  const wx = -uy;
  const wy = ux;

  // Eye center is midpoint between inner and outer canthus
  const centerX = (leftCorner.x + rightCorner.x) / 2;
  const centerY = (leftCorner.y + rightCorner.y) / 2;

  const vx = iris.x - centerX;
  const vy = iris.y - centerY;

  // Project iris offset onto roll-invariant eye basis, normalized by eyeWidth
  const dx = (vx * ux + vy * uy) / eyeWidth;
  const dy = (vx * wx + vy * wy) / eyeWidth;

  // In mirrored webcam view:
  // Looking screen-right decreases camera-x (dx < 0), so 0.5 - dx * 2.5 maps to > 0.5.
  const hRatio = Math.max(0, Math.min(1, 0.5 - dx * 2.5));
  const vRatio = Math.max(0, Math.min(1, 0.5 + dy * 3.0));

  return { hRatio, vRatio, isOpen: true, openness };
}

export function computeIrisGaze(
  landmarks: NormalizedLandmark[],
  previousGaze?: IrisGazeResult
): IrisGazeResult {
  if (!landmarks || landmarks.length < 478) {
    return {
      hRatio: previousGaze?.hRatio ?? 0.5,
      vRatio: previousGaze?.vRatio ?? 0.5,
      isBlinking: true,
      rightOpenness: 0,
      leftOpenness: 0,
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
    previousGaze ? { h: previousGaze.hRatio, v: previousGaze.vRatio } : undefined
  );

  // Left eye: inner (362) is image-left, outer (263) is image-right
  const left = calculateSingleEyeGaze(
    landmarks,
    LANDMARKS.LEFT_EYE.INNER,
    LANDMARKS.LEFT_EYE.OUTER,
    LANDMARKS.LEFT_EYE.UPPER,
    LANDMARKS.LEFT_EYE.LOWER,
    LANDMARKS.LEFT_EYE.IRIS,
    previousGaze ? { h: previousGaze.hRatio, v: previousGaze.vRatio } : undefined
  );

  const isBlinking = !right.isOpen && !left.isOpen;

  if (isBlinking && previousGaze) {
    return {
      hRatio: previousGaze.hRatio,
      vRatio: previousGaze.vRatio,
      isBlinking: true,
      rightOpenness: right.openness,
      leftOpenness: left.openness,
    };
  }

  let hRatio = 0.5;
  let vRatio = 0.5;

  if (right.isOpen && left.isOpen) {
    hRatio = (right.hRatio + left.hRatio) / 2;
    vRatio = (right.vRatio + left.vRatio) / 2;
  } else if (right.isOpen) {
    hRatio = right.hRatio;
    vRatio = right.vRatio;
  } else if (left.isOpen) {
    hRatio = left.hRatio;
    vRatio = left.vRatio;
  } else if (previousGaze) {
    hRatio = previousGaze.hRatio;
    vRatio = previousGaze.vRatio;
  }

  return {
    hRatio,
    vRatio,
    isBlinking,
    rightOpenness: right.openness,
    leftOpenness: left.openness,
  };
}
