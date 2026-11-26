import { NormalizedLandmark } from '../src/lib/math/iris-gaze';

export function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(rand: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
}

export function mouthLandmarks(
  mar: number,
  w: number,
  h: number,
  mouthWidthPx = 60
): NormalizedLandmark[] {
  const lm: NormalizedLandmark[] = [];
  const cx = 0.5;
  const cy = 0.6;
  const gapPx = mar * mouthWidthPx;
  lm[61] = { x: cx - mouthWidthPx / 2 / w, y: cy, z: 0 };
  lm[291] = { x: cx + mouthWidthPx / 2 / w, y: cy, z: 0 };
  lm[13] = { x: cx, y: cy - gapPx / 2 / h, z: 0 };
  lm[14] = { x: cx, y: cy + gapPx / 2 / h, z: 0 };
  return lm;
}

export function speechLikeMar(durationS: number, seed: number): (tS: number) => number {
  const rand = mulberry32(seed);
  const pulses: Array<{ t: number; peak: number }> = [];
  let t = 0.1;
  while (t < durationS) {
    pulses.push({ t, peak: 0.08 + rand() * 0.2 });
    t += 0.15 + rand() * 0.15 + (rand() < 0.12 ? 0.5 : 0);
  }
  const half = 0.09;
  return (tS) => {
    let v = 0.03;
    for (const p of pulses) {
      const d = Math.abs(tS - p.t);
      if (d < half) {
        v = Math.max(v, 0.03 + p.peak * 0.5 * (1 + Math.cos((Math.PI * d) / half)));
      }
    }
    return v;
  };
}
