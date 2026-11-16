// Synthetic export: 300 frames @25 fps, 6 slots (3 home, 3 away), a pass from home #1 to away #4
// between frames 100 and 130, ball lost 60..69, slot 5 vanishes 150..199, slot 2 jumps 8 m at 210.
import { DataSet } from '../../src/data.js';

export const FPS = 25;
export function makeMeta(over = {}) {
  const n = 300, S = 6;
  return {
    source: 'tactical', fps: FPS, start: 1000, end: 1000 + n - 1, n, width: 1920, height: 1080,
    pitch: [105.3, 68.0], block: 9 + 6 + 4 * S, n_slots: S,
    slots: [1, 2, 3, 4, 5, 6].map(k => ({ id: String(k), num: String(k), team: k <= 3 ? 'H' : 'A' })),
    home_dir: 1, halves: { 1: [0, 5000], 2: [6000, 12000] },
    gt: 'H'.repeat(120) + 'A'.repeat(100) + 'D'.repeat(40) + 'H'.repeat(40),
    pred: null, tags: [], flags: [],
    events: { kick: [100], bjump: [], pjump: [[210, 1, 8.0]], vanish: [[150, 4]], appear: [[200, 4]] },
    count: new Array(n).fill(6).map((c, r) => (r >= 150 && r < 200 ? 5 : c)),
    dropout: new Array(n).fill(0), pmax: new Array(n).fill(4.0), hom_err: new Array(n).fill(0.5),
    has_z: false, has_ball_speed: false, bolt: 12.4, ball_radius: 0.11, ...over,
  };
}
export function makeData(meta) {
  const { n, block, n_slots: S } = meta;
  const D = new Float32Array(n * block).fill(NaN);
  // identity-ish homography: pitch (0..1) -> pixels (x*1000+100, y*600+100)
  const H = [1000, 0, 100, 0, 600, 100, 0, 0, 1];
  const base = [[0.30, 0.50], [0.20, 0.30], [0.25, 0.70], [0.60, 0.50], [0.70, 0.30], [0.75, 0.70]];
  for (let r = 0; r < n; r++) {
    const o = r * block;
    for (let i = 0; i < 9; i++) D[o + i] = H[i];
    // ball: carried by slot 0 until 100, flies to slot 3 (x 0.30 -> 0.60) over 100..130, then carried
    let bx, by;
    if (r < 100) { bx = 0.30 + 0.005 / 105.3 * r; by = 0.50; }
    else if (r <= 130) { bx = 0.30 + 0.30 * (r - 100) / 30; by = 0.50; }
    else { bx = 0.60 + 0.003 * (r - 130) / 105.3; by = 0.50; }
    if (!(r >= 60 && r < 70)) { D[o + 9] = bx; D[o + 10] = by; D[o + 11] = NaN; D[o + 12] = NaN; D[o + 13] = r > 100 && r <= 130 ? 26 : 1.0; D[o + 14] = r === 100 ? 20 : 0; }
    for (let k = 0; k < S; k++) {
      if (k === 4 && r >= 150 && r < 200) continue;            // vanished
      let [x, y] = base[k];
      if (k === 0 && r < 100) x = bx - 0.004;                    // carrier
      if (k === 3 && r > 130) x = bx + 0.004;                    // receiver
      if (k === 1 && r === 210) x += 8 / 105.3;                  // jump
      const p = o + 15 + 4 * k; D[p] = x; D[p + 1] = y; D[p + 2] = 1.5; D[p + 3] = 1.4;
    }
  }
  return D;
}
export function makeDataSet(over) { const m = makeMeta(over); return new DataSet(m, makeData(m)); }
export class MemoryStorage { constructor() { this.m = new Map(); } get(k) { return this.m.get(k) ?? null; } set(k, v) { this.m.set(k, v); } }

// A filtered ball for the same 300 frames: the detector point, except during the lost stretch
// (where the filter keeps the ball on the carrier) and an "air ball" excursion the filter rejects.
export function makeBallq(meta) {
  const n = meta.n, Q = new Float32Array(n * 8).fill(NaN), D = makeData(meta);
  let lhRaw = '', lhPost = '';
  for (let r = 0; r < n; r++) {
    const o = r * 8, bx = D[r * meta.block + 9];
    const x = Number.isFinite(bx) ? bx : 0.30 + 0.005 / 105.3 * r;           // lost stretch: carried on
    Q[o] = x; Q[o + 1] = 0.50; Q[o + 2] = Number.isFinite(bx) ? 0.4 / 105.3 : 1.2 / 105.3; Q[o + 3] = 0.4 / 68;
    Q[o + 4] = 4; Q[o + 5] = Number.isFinite(bx) ? 0.95 : 0; Q[o + 6] = 0.8; Q[o + 7] = Number.isFinite(bx) ? 0 : 1;
    lhRaw += r < 110 ? 'H' : r < 220 ? 'A' : r < 260 ? 'X' : 'H';
    lhPost += r < 120 ? 'H' : r < 220 ? 'A' : r < 260 ? 'X' : 'H';                 // the filtered one keeps H to 120, as GT does
  }
  return { Q, ballq: { lh_raw: lhRaw, lh_post: lhPost, moved: [[60, 2.5]], quality: { config_hash: 'abc', white_team: 'A', gt_agreement_lh_raw: 0.9, gt_agreement_lh_posterior: 0.95 } } };
}
export function makeDataSetWithBallq(over) { const m = makeMeta(over); const b = makeBallq(m); m.ballq = b.ballq; return new DataSet(m, makeData(m), b.Q); }
