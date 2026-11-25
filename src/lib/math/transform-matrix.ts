export interface EulerAngles {
  pitch: number; // Rotation around X (nodding up/down). Positive is looking up, negative is looking down.
  yaw: number;   // Rotation around Y (turning left/right).
  roll: number;  // Rotation around Z (tilting head side-to-side).
}

/**
 * Decomposes MediaPipe's 4x4 Metric 3D facial transformation matrix into Euler angles in degrees.
 * 
 * MediaPipe's matrix is provided as a flat 16-element column-major Float32Array or number array:
 * [ m00, m10, m20, m30,
 *   m01, m11, m21, m31,
 *   m02, m12, m22, m32,
 *   m03, m13, m23, m33 ]
 * 
 * Therefore: R[row][col] = data[col * 4 + row]
 */
export function decomposeTransformMatrix(data: number[] | Float32Array): EulerAngles {
  if (!data || data.length < 16) {
    return { pitch: 0, yaw: 0, roll: 0 };
  }

  // Extract the 3x3 rotation submatrix: R[row][col]
  const R00 = data[0 * 4 + 0];
  const R10 = data[0 * 4 + 1];
  const R20 = data[0 * 4 + 2];

  const R01 = data[1 * 4 + 0];
  const R11 = data[1 * 4 + 1];
  const R21 = data[1 * 4 + 2];

  const R02 = data[2 * 4 + 0];
  const R12 = data[2 * 4 + 1];
  const R22 = data[2 * 4 + 2];

  // Pitch (rotation around X):
  // R21 is -sin(pitch). Clamp strictly to [-1, 1] to avoid NaN from floating-point inaccuracies.
  const sinPitch = Math.max(-1, Math.min(1, -R21));
  const pitch = Math.asin(sinPitch) * (180 / Math.PI);

  // Yaw (rotation around Y): atan2(R20, R22)
  const yaw = Math.atan2(R20, R22) * (180 / Math.PI);

  // Roll (rotation around Z): atan2(R01, R11)
  const roll = Math.atan2(R01, R11) * (180 / Math.PI);

  return { pitch, yaw, roll };
}
