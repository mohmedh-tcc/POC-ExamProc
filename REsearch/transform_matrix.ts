export interface EulerAngles {
  pitch: number; // Rotation around X (nodding up/down). Positive is looking up, negative is looking down.
  yaw: number;   // Rotation around Y (turning left/right).
  roll: number;  // Rotation around Z (tilting head side-to-side).
}

/** Row-major 3x3 rotation matrix: [R00, R01, R02, R10, R11, R12, R20, R21, R22] */
export type Mat3 = [
  number, number, number,
  number, number, number,
  number, number, number
];

/** 3D vector [x, y, z] in metric camera space (cm) or unit direction space */
export type Vec3 = [number, number, number];

export interface RigidTransform3D {
  /** Orthonormal 3x3 rotation matrix R_head mapping head-local vectors to camera/world space */
  rotation: Mat3;
  /** 3D translation vector t_head = [tx, ty, tz] in MediaPipe metric units (cm; tz < 0 in front of camera) */
  translation: Vec3;
  /** Decomposed Euler angles in degrees (for UI gauges & safety guardrails) */
  euler: EulerAngles;
  /** True if the input 4x4 matrix was present and well-conditioned */
  isValid: boolean;
}

const IDENTITY_MAT3: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

/** Multiply a row-major 3x3 matrix by a 3D vector: R * v */
export function mat3MulVec3(R: Mat3, v: Vec3): Vec3 {
  return [
    R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
    R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
    R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
  ];
}

/** Normalize a 3D vector; returns [0, 0, 1] if near-zero */
export function normalizeVec3(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-8) return [0, 0, 1];
  return [v[0] / len, v[1] / len, v[2] / len];
}

/**
 * Extracts the full 3D rigid transform (R_head, t_head) and Euler angles from
 * MediaPipe's 4x4 column-major Metric 3D facial transformation matrix:
 *
 *   [ m00, m10, m20, m30,
 *     m01, m11, m21, m31,
 *     m02, m12, m22, m32,
 *     m03, m13, m23, m33 ]
 *
 * Therefore: M[row][col] = data[col * 4 + row].
 * Translation t_head = [data[12], data[13], data[14]] (in cm; tz < 0 in front of camera).
 */
export function extractRigidTransform(data?: number[] | Float32Array | null): RigidTransform3D {
  if (!data || data.length < 16) {
    return {
      rotation: [...IDENTITY_MAT3] as Mat3,
      translation: [0, 0, -55],
      euler: { pitch: 0, yaw: 0, roll: 0 },
      isValid: false,
    };
  }

  // Column vectors of the 3x3 upper-left rotation block
  let c0: Vec3 = [data[0], data[1], data[2]];
  let c1: Vec3 = [data[4], data[5], data[6]];
  let c2: Vec3 = [data[8], data[9], data[10]];

  // Orthonormalize via modified Gram-Schmidt to guard against scale/numerical drift
  c0 = normalizeVec3(c0);
  const dot01 = c0[0] * c1[0] + c0[1] * c1[1] + c0[2] * c1[2];
  c1 = normalizeVec3([c1[0] - dot01 * c0[0], c1[1] - dot01 * c0[1], c1[2] - dot01 * c0[2]]);
  // Right-handed cross product c2 = c0 x c1 guarantees det(R) = +1
  c2 = [
    c0[1] * c1[2] - c0[2] * c1[1],
    c0[2] * c1[0] - c0[0] * c1[2],
    c0[0] * c1[1] - c0[1] * c1[0],
  ];

  // Store row-major R[row][col]
  const R: Mat3 = [
    c0[0], c1[0], c2[0],
    c0[1], c1[1], c2[1],
    c0[2], c1[2], c2[2],
  ];

  const tx = Number.isFinite(data[12]) ? data[12] : 0;
  const ty = Number.isFinite(data[13]) ? data[13] : 0;
  // MediaPipe metric z is negative in front of the camera (typically -35 to -80 cm)
  const tz = Number.isFinite(data[14]) && Math.abs(data[14]) > 5 ? data[14] : -55;

  const sinPitch = Math.max(-1, Math.min(1, -R[7])); // -R21
  const pitch = Math.asin(sinPitch) * RAD_TO_DEG;
  const yaw = Math.atan2(R[6], R[8]) * RAD_TO_DEG;   // atan2(R20, R22)
  const roll = Math.atan2(R[1], R[4]) * RAD_TO_DEG;  // atan2(R01, R11)

  return {
    rotation: R,
    translation: [tx, ty, tz],
    euler: { pitch, yaw, roll },
    isValid: true,
  };
}

/**
 * Reconstructs a 3x3 head rotation matrix R_head from decomposed Euler angles
 * matching the exact convention of `decomposeTransformMatrix` (Rz(roll) * Rx(pitch) * Ry(-yaw)).
 * Used as a fallback when only EulerAngles (not the raw 4x4 matrix) are available.
 */
export function eulerToRotationMatrix(euler: EulerAngles): Mat3 {
  const p = euler.pitch * DEG_TO_RAD;
  // In decomposeTransformMatrix, yaw = atan2(R20, R22), where Ry(theta) has R20 = -sin(theta),
  // so the right-handed rotation angle around +Y is -yaw.
  const y = -euler.yaw * DEG_TO_RAD;
  const r = euler.roll * DEG_TO_RAD;

  const cp = Math.cos(p), sp = Math.sin(p);
  const cy = Math.cos(y), sy = Math.sin(y);
  const cr = Math.cos(r), sr = Math.sin(r);

  // R = Rz(r) * Rx(p) * Ry(y)
  return [
    cr * cy - sr * sp * sy,  sr * cp,  cr * sy + sr * sp * cy,
    -sr * cy - cr * sp * sy, cr * cp, -sr * sy + cr * sp * cy,
    -cp * sy,                -sp,      cp * cy,
  ];
}

/**
 * Backwards-compatible wrapper that decomposes MediaPipe's 4x4 Metric 3D
 * facial transformation matrix into Euler angles in degrees.
 */
export function decomposeTransformMatrix(data: number[] | Float32Array): EulerAngles {
  return extractRigidTransform(data).euler;
}