// Turns per frame values into a one pixel high strip of RGB bytes, W pixels wide.
// No canvas involved, so this can be tested in node. They all take (..., r0, r1, W, ...)
// and give back a Uint8Array of W*3.

const bucket = (r0, r1, W) => { const span = r1 - r0 + 1; return i => [r0 + Math.floor(i * span / W), Math.max(r0 + Math.floor(i * span / W) + 1, r0 + Math.floor((i + 1) * span / W))]; };
const put = (px, i, c) => { px[i * 3] = c[0]; px[i * 3 + 1] = c[1]; px[i * 3 + 2] = c[2]; };

export function rasterLabel(get, r0, r1, W, cmap, bg) {
  const px = new Uint8Array(W * 3), B = bucket(r0, r1, W);
  for (let i = 0; i < W; i++) {
    const [a, b] = B(i); const cnt = new Map();
    for (let r = a; r < b; r++) { const l = get(r) || ''; cnt.set(l, (cnt.get(l) || 0) + 1); }
    let best = '', bc = -1; for (const [l, c] of cnt) if (c > bc) { bc = c; best = l; }
    put(px, i, best && best !== 'X' && cmap[best] ? cmap[best] : bg);
  }
  return px;
}
export function rasterMask(get, r0, r1, W, color, bg) {
  const px = new Uint8Array(W * 3), B = bucket(r0, r1, W);
  for (let i = 0; i < W; i++) { const [a, b] = B(i); let on = false; for (let r = a; r < b; r++) if (get(r)) { on = true; break; } put(px, i, on ? color : bg); }
  return px;
}
export function rasterIntensity(get, r0, r1, W, vmax, color, bg, { bad = null, badColor = [240, 60, 60] } = {}) {
  const px = new Uint8Array(W * 3), B = bucket(r0, r1, W);
  for (let i = 0; i < W; i++) {
    const [a, b] = B(i); let v = -1;
    for (let r = a; r < b; r++) { const x = get(r); if (Number.isFinite(x) && x > v) v = x; }
    if (v < 0) { put(px, i, bg); continue; }
    if (bad !== null && v > bad) { put(px, i, badColor); continue; }
    const k = Math.min(1, v / vmax); put(px, i, [bg[0] + (color[0] - bg[0]) * k, bg[1] + (color[1] - bg[1]) * k, bg[2] + (color[2] - bg[2]) * k]);
  }
  return px;
}
export function rasterCategorical(get, r0, r1, W, bg) {
  // get(r) gives a colour or null, the first colour inside a pixel wins
  const px = new Uint8Array(W * 3), B = bucket(r0, r1, W);
  for (let i = 0; i < W; i++) { const [a, b] = B(i); let c = null; for (let r = a; r < b; r++) { c = get(r); if (c) break; } put(px, i, c || bg); }
  return px;
}
export function rasterInterval(items, r0, r1, W, bg) {
  // items look like {f, f2, color}; the whole span gets painted
  const px = new Uint8Array(W * 3); for (let i = 0; i < W; i++) put(px, i, bg);
  const span = r1 - r0 + 1;
  for (const it of items) {
    const a = Math.max(it.f, r0), b = Math.min(it.f2 ?? it.f, r1); if (b < a) continue;
    const i0 = Math.floor((a - r0) / span * W), i1 = Math.max(i0, Math.floor((b - r0) / span * W));
    for (let i = i0; i <= Math.min(W - 1, i1); i++) put(px, i, it.color);
  }
  return px;
}
export function rasterSlot(runs, jumps, r0, r1, W, presentColor, jumpColor, bg) {
  const px = rasterInterval(runs.filter(x => x.present).map(x => ({ f: x.a, f2: x.b, color: presentColor })), r0, r1, W, bg);
  const span = r1 - r0 + 1;
  for (const j of jumps) if (j >= r0 && j <= r1) { const i = Math.floor((j - r0) / span * W); put(px, i, jumpColor); if (i + 1 < W) put(px, i + 1, jumpColor); }
  return px;
}
// How many players are usually tracked in this export, the most common count.
export function usualCount(counts) {
  const freq = new Map(); for (const c of counts) freq.set(c, (freq.get(c) || 0) + 1);
  let best = 0, bc = -1; for (const [c, f] of freq) if (f > bc) { bc = f; best = c; } return best;
}
// Per bin summaries over the whole export for the overview bands.
// restarts and kicks are counts, everything else is a fraction of the bin.
export function aggregateBands(ds, events, poss, binFrames) {
  const out = []; const n = ds.n, m = ds.meta;
  const usual = m.count ? usualCount(m.count) : 22;
  for (let a = 0; a < n; a += binFrames) {
    const b = Math.min(n, a + binFrames); let lost = 0, h = 0, aa = 0, cnt = 0, hom = 0;
    for (let r = a; r < b; r++) {
      if (!ds.ball(r)) lost++;
      if (poss.lh[r] === 'H') h++; else if (poss.lh[r] === 'A') aa++;
      if (m.count && m.count[r] !== usual) cnt++;
      if (m.hom_err && (m.hom_err[r] < 0 || m.hom_err[r] > 3)) hom++;
    }
    const len = b - a;
    out.push({ a, b: b - 1, len,
      restarts: events.restartFrames.filter(r => r >= a && r < b).length,
      kicks: [...events.kick].filter(r => r >= a && r < b).length,
      lostFrac: lost / len, shareH: h / len, shareA: aa / len, countBad: cnt / len, homBad: hom / len });
  }
  return out;
}
