// Read only access to one export (meta.json plus data.bin).
// r is the frame index inside the export, 0..n-1. The video frame is start + r.
// One frame in data.bin is a block of floats: 9 for the homography, 6 for the ball
// (x, y, z, speed from the data, speed from frames, acceleration), then 4 per slot
// (x, y, speed from the data, speed from frames).
// A second optional file, ballq.bin, holds the filtered ball from the ballq pipeline: 8 floats per
// frame (mean x, mean y, sd x, sd y, entropy bits, trust tau, holder p, flag bits).

export class DataSet {
  constructor(meta, f32, bq = null) {
    this.meta = meta; this.D = f32; this.Q = bq;
    this.n = meta.n; this.fps = meta.fps; this.start = meta.start; this.end = meta.end;
    this.S = meta.n_slots; this.block = meta.block; this.pitch = meta.pitch;
    this.slots = meta.slots;
    this.teamNames = meta.team_names || { H: 'home', A: 'away' };
  }
  hasBallq() { return !!(this.Q && this.meta.ballq); }
  // The probability map. grid.bin holds, per frame, cells of (uint16 row, uint16 col, uint16 mass x 65535);
  // grid_index.bin has the byte offset of each frame. One frame is fetched with a Range request when it is
  // first looked at and kept in a small cache, so a two-hour match costs nothing until you seek into it.
  hasGrid() { return !!(this.meta.ballq && this.meta.ballq.grid && this.gridOffsets); }
  setGridIndex(url, offsets, onArrive) { this.gridUrl = url; this.gridOffsets = offsets; this.gridCache = new Map(); this.gridPending = new Set(); this.onGridArrive = onArrive; }
  gridCells(r) {
    if (!this.hasGrid() || r < 0 || r >= this.n) return null;
    const hit = this.gridCache.get(r);
    if (hit) return hit;
    const a = this.gridOffsets[r], b = this.gridOffsets[r + 1];
    if (b <= a) return { n: 0 };
    if (!this.gridPending.has(r)) {
      this.gridPending.add(r);
      fetch(this.gridUrl, { headers: { Range: `bytes=${a}-${b - 1}` } }).then(res => res.arrayBuffer()).then(buf => {
        const cells = this.decodeGrid(new Uint16Array(buf));
        if (this.gridCache.size > 3000) this.gridCache.delete(this.gridCache.keys().next().value);
        this.gridCache.set(r, cells); this.gridPending.delete(r);
        if (this.onGridArrive) this.onGridArrive(r);
      }).catch(() => this.gridPending.delete(r));
    }
    return null;
  }
  decodeGrid(u16) {
    const n = u16.length / 3, i = new Uint16Array(n), j = new Uint16Array(n), p = new Float32Array(n); let max = 0;
    for (let k = 0; k < n; k++) { i[k] = u16[3 * k]; j[k] = u16[3 * k + 1]; p[k] = u16[3 * k + 2] / 65535; if (p[k] > max) max = p[k]; }
    return { n, i, j, p, max };
  }
  // pitch-unit corners of grid cell (i, j)
  cellBox(i, j) { const g = this.meta.ballq.grid, cw = g.cell_m / this.pitch[0], ch = g.cell_m / this.pitch[1]; return [j * cw, i * ch, (j + 1) * cw, (i + 1) * ch]; }
  // the filtered ball: mean and spread in pitch units, plus how much the detector was trusted
  ballq(r) {
    if (!this.Q || r < 0 || r >= this.n) return null;
    const o = r * 8, Q = this.Q;
    return Number.isFinite(Q[o]) ? { x: Q[o], y: Q[o + 1], sdx: Q[o + 2], sdy: Q[o + 3], ent: Q[o + 4], tau: Q[o + 5], hp: Q[o + 6], flags: Q[o + 7] } : null;
  }
  lhRaw(r) { return this.meta.ballq ? this.meta.ballq.lh_raw[r] : null; }
  // our dead-ball detector and the label the submission would carry: D when dead, else L&H on the filtered ball
  // the mask comes from export/dead_ball.json through EventIndex (app.js hands it over after deriving events)
  get deadMask() { return this._deadMask || null; }
  set deadMask(m) { this._deadMask = m || null; this._sub = null; }
  hasDeadOurs() { return !!this.deadMask; }
  deadOurs(r) { return this.deadMask ? !!this.deadMask[r] : false; }
  // D when we say dead, else L&H on the filtered ball, then runs under minSpellS bordered by the same team are
  // absorbed, exactly as ballq/smooth.py does for the CSV (the scorer counts possessions, so a blip costs twice)
  submission(r) { if (!this._sub) this._sub = this.buildSubmission(); return this._sub[r] || null; }
  buildSubmission(minSpellS = 1.0) {
    const n = this.n, lab = new Array(n);
    for (let r = 0; r < n; r++) { if (this.deadOurs(r)) lab[r] = 'D'; else { const t = this.lhPost(r); lab[r] = t && t !== 'X' ? t : 'X'; } }
    const minLen = Math.round(minSpellS * this.fps);
    for (let round = 0; round < 3; round++) {
      const runs = []; let s = 0;
      for (let i = 1; i <= n; i++) if (i === n || lab[i] !== lab[s]) { runs.push([s, i, lab[s]]); s = i; }
      let changed = false;
      runs.forEach(([a, b, v], k) => {
        if ((v !== 'H' && v !== 'A') || b - a >= minLen) return;
        let prev = null; for (let j = k - 1; j >= 0; j--) if (runs[j][2] !== 'X') { prev = runs[j][2]; break; }
        let nxt = null; for (let j = k + 1; j < runs.length; j++) if (runs[j][2] !== 'X') { nxt = runs[j][2]; break; }
        if ((prev === 'H' || prev === 'A') && prev === nxt && prev !== v) { for (let i = a; i < b; i++) lab[i] = prev; changed = true; }
      });
      if (!changed) break;
    }
    return lab.map(v => (v === 'X' ? null : v));
  }
  resetSubmission() { this._sub = null; }
  lhPost(r) { return this.meta.ballq ? this.meta.ballq.lh_post[r] : null; }
  // how far the filter moved the ball from the detector point, in metres (NaN when either is missing)
  ballMoved(r) { const b = this.ball(r), q = this.ballq(r); return (b && q) ? this.metres(b, q) : NaN; }
  H(r) { const o = r * this.block; return Number.isFinite(this.D[o]) ? this.D.subarray(o, o + 9) : null; }
  ball(r) {
    if (r < 0 || r >= this.n) return null;
    const o = r * this.block + 9, D = this.D;
    return Number.isFinite(D[o]) ? { x: D[o], y: D[o + 1], z: D[o + 2], sd: D[o + 3], sf: D[o + 4], acc: D[o + 5] } : null;
  }
  slot(r, k) {
    if (r < 0 || r >= this.n) return null;
    const o = r * this.block + 15 + 4 * k, D = this.D;
    return Number.isFinite(D[o]) ? { x: D[o], y: D[o + 1], sd: D[o + 2], sf: D[o + 3], k } : null;
  }
  players(r) { const out = []; for (let k = 0; k < this.S; k++) { const p = this.slot(r, k); if (p) out.push(p); } return out; }
  present(r, k) { return Number.isFinite(this.D[r * this.block + 15 + 4 * k]); }
  sideOf(k) { return this.slots[k].team; }
  numOf(k) { return this.slots[k].num; }
  metres(a, b) { return Math.hypot((a.x - b.x) * this.pitch[0], (a.y - b.y) * this.pitch[1]); }
  proj(Hm, x, y) { const w = Hm[6] * x + Hm[7] * y + Hm[8]; return [(Hm[0] * x + Hm[1] * y + Hm[2]) / w, (Hm[3] * x + Hm[4] * y + Hm[5]) / w]; }
  inv3(h) {
    const [a, b, c, d, e, f, g, hh, i] = h;
    const A = e * i - f * hh, B = -(d * i - f * g), C = d * hh - e * g, det = a * A + b * B + c * C;
    return [A / det, -(b * i - c * hh) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, -(a * f - c * d) / det, C / det, -(a * hh - b * g) / det, (a * e - b * d) / det];
  }
  pitchFromPx(r, px, py) { const Hm = this.H(r); if (!Hm) return null; const p = this.proj(this.inv3(Hm), px, py); return { x: p[0], y: p[1] }; }
  halfOf(f) { for (const h in this.meta.halves) { const [a, b] = this.meta.halves[h]; if (f >= a && f <= b) return +h; } return 0; }
  attackDir(f) { const hv = this.meta.halves; if (!this.meta.home_dir || !hv || !hv['2']) return 0; return f < hv['2'][0] ? this.meta.home_dir : -this.meta.home_dir; }
  gt(r) { return this.meta.gt ? this.meta.gt[r] : null; }
  teamName(s) { return this.teamNames[s] || s; }
  pred(r) { return this.meta.pred ? this.meta.pred[r] : null; }
}
