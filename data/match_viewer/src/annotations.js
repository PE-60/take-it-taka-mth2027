// Everything the user writes down lives here. No DOM, no localStorage: whoever creates the store
// hands in something with get(key) and set(key, value).
//
// Three kinds of things, one vocabulary:
//   segment   {f, team, state}   my possession reading, runs until the next segment
//   point     {f, cat, type, ...} an event or a data error at one frame
//   interval  a point with an f2, for data errors that last a while
// Layer A is mine and editable, layer B is someone else's import and read only.

export const SCHEMA = {
  state: { segment: true, items: [
    ['controlled', '1', 'controlled possession', '#5adc5a'], ['flight', '2', 'pass / ball in flight', '#78dcff'],
    ['loose', '3', 'loose ball / contested', '#f5d73c'], ['dead', '4', 'dead ball', '#bdbdbd'], ['clear', '0', 'clear from here', '#4a4e55']] },
  event: { items: [
    ['pass', 'P', 'pass', '#5adc5a'], ['shot', 'S', 'shot', '#ff8c3c'], ['tackle', 'T', 'tackle / duel', '#f5d73c'],
    ['interception', 'I', 'interception', '#ff5a5a'], ['clearance', 'C', 'clearance', '#c8ff78'], ['deflection', 'J', 'deflection (no control)', '#dcdcdc'],
    ['foul', 'F', 'foul', '#ff3cb4'], ['out', 'O', 'ball out of play', '#ff3c3c'], ['restart', 'K', 'restart taken', '#ffffff']] },
  data: { interval: true, items: [
    ['ball_pos', 'N', 'true ball position', '#ffe600', 'pitch'], ['player_missing', 'M', 'player missing', '#ff8cc8', 'pitch+team'],
    ['ghost', 'G', 'ghost / duplicate marker', '#c850ff', 'player'], ['wrong_team', 'W', 'wrong team for marker', '#ff5a5a', 'player'],
    ['id_swap', 'V', 'identity swap', '#c850ff', 'player2'], ['lag', 'Y', 'marker lags the real player', '#ffb45a', 'player'],
    ['homography_off', 'U', 'homography misaligned', '#f03c3c'], ['gt_doubt', 'R', 'GT looks wrong here', '#ffe600'], ['unclear', 'Q', 'cannot tell from video', '#9aa0a8']] },
};
export const TOOL = {};
for (const cat in SCHEMA) for (const [type, key, label, color, need] of SCHEMA[cat].items) TOOL[key] = { cat, type, label, color, need: need || null, interval: !!SCHEMA[cat].interval };
export const TYPE = Object.fromEntries(Object.values(TOOL).map(t => [t.type, t]));
const ALIVE = new Set(['controlled', 'flight', 'loose']);
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

export class AnnotationStore {
  constructor(storage, key, ds, { undoDepth = 50 } = {}) {
    this.storage = storage; this.key = key; this.ds = ds; this.undoDepth = undoDepth;
    this.segments = []; this.points = []; this.layerB = { segments: [], points: [] }; this.bookmarks = []; this.queues = {}; this.rows = null; this.overview = null;
    this.pending = null; this._undo = [];
    this._load();
  }
  // ---------------- persistence
  _load() {
    let raw = null; try { raw = JSON.parse(this.storage.get(this.key) || 'null'); } catch (e) { raw = null; }
    if (!raw) return;
    this.segments = raw.segments || []; this.points = raw.points || raw.ann || []; this.bookmarks = raw.bookmarks || [];
    this.layerB = raw.layerB || { segments: [], points: [] }; this.queues = raw.queues || {}; this.rows = raw.rows ?? null; this.overview = raw.overview ?? null;
    if (raw.labels && !raw.segments) {                   // old style MINE labels from before, turn them into segments
      for (const [f, l] of Object.entries(raw.labels)) this.segments.push(l === 'X' ? { f: +f, team: null, state: 'clear' } : l === 'D' ? { f: +f, team: null, state: 'dead' } : { f: +f, team: l, state: 'controlled' });
      this.segments.sort((a, b) => a.f - b.f); this._save();
    }
  }
  _save() { this.storage.set(this.key, JSON.stringify({ segments: this.segments, points: this.points, bookmarks: this.bookmarks, layerB: this.layerB, queues: this.queues, rows: this.rows, overview: this.overview })); }
  _snapshot() { this._undo.push(JSON.stringify({ s: this.segments, p: this.points })); if (this._undo.length > this.undoDepth) this._undo.shift(); }
  undo() { const s = this._undo.pop(); if (!s) return false; const o = JSON.parse(s); this.segments = o.s; this.points = o.p; this._save(); return true; }

  // ---------------- segments (S4)
  setSegment(f, change) {
    this._snapshot();
    const prev = this.segmentAt(f, true);               // look before removing, so a second key on the same frame keeps what was there
    this.segments = this.segments.filter(s => s.f !== f);
    if (change === null) this.segments.push({ f, team: null, state: 'clear' });
    else {
      let team = change.team !== undefined ? change.team : prev.team;
      let state = change.state !== undefined ? change.state : (prev.state === 'clear' || prev.state === 'dead' ? 'controlled' : prev.state);
      if (change.team !== undefined && (prev.state === 'dead' || prev.state === 'clear')) state = 'controlled';
      if (state === 'dead') team = null;
      this.segments.push({ f, team, state });
    }
    this.segments.sort((a, b) => a.f - b.f); this._save();
  }
  segmentAt(r, raw = false, layer = 'A') {
    const L = layer === 'B' ? this.layerB.segments : this.segments;
    let best = null; for (const s of L) { if (s.f <= r) best = s; else break; }
    if (!best) return raw ? { team: null, state: 'clear' } : { team: null, state: 'clear' };
    return { team: best.team, state: best.state };
  }
  derivedLabel(r, layer = 'A') { const s = this.segmentAt(r, false, layer); if (s.state === 'dead') return 'D'; if (ALIVE.has(s.state) && s.team) return s.team; return 'X'; }

  // ---------------- points and intervals (S2)
  addPoint(f, a) { this._snapshot(); const it = { id: uid(), f, ...a }; this.points.push(it); this.points.sort((p, q) => p.f - q.f); this._save(); return it; }
  remove(id) { this._snapshot(); this.points = this.points.filter(p => p.id !== id); this._save(); }
  beginInterval(f, a) { this.pending = { id: uid(), f, ...a }; return this.pending; }
  endInterval(f2) {
    if (!this.pending) return null; const p = this.pending; this.pending = null;
    const a = Math.min(p.f, f2), b = Math.max(p.f, f2); p.f = a; p.f2 = b;
    this._snapshot(); this.points.push(p); this.points.sort((x, y) => x.f - y.f); this._save(); return p;
  }
  discardPending() { this.pending = null; }
  nudgePending(dxM, dyM) { if (!this.pending || this.pending.x === undefined) return; this.pending.x += dxM / this.ds.pitch[0]; this.pending.y += dyM / this.ds.pitch[1]; }
  durationS(a) { return ((a.f2 ?? a.f) - a.f + 1) / this.ds.fps; }
  items(layer = 'A') { return layer === 'B' ? this.layerB.points : this.points; }
  at(r, layer = 'A') { return this.items(layer).filter(a => a.f <= r && r <= (a.f2 ?? a.f)); }
  near(r, tolFrames, layer = 'A') { return this.items(layer).filter(a => a.f - tolFrames <= r && r <= (a.f2 ?? a.f) + tolFrames); }

  // ---------------- bookmarks
  addBookmark(f, note, extra = {}) { this.bookmarks.push({ f, note, ...extra }); this.bookmarks.sort((a, b) => a.f - b.f); this._save(); }
  removeBookmark(i) { this.bookmarks.splice(i, 1); this._save(); }

  // ---------------- queues (S1) and row filter
  saveQueue(name, obj) { this.queues[name] = obj; this._save(); }
  savedQueue(name) { return this.queues[name] || null; }
  setRows(rows) { this.rows = rows; this._save(); }
  setOverview(names) { this.overview = names; this._save(); }

  // ---------------- layer B and agreement (S3)
  importLayer(json, layer = 'B') {
    const toRel = f => f;                                // frames in a layer file are already relative to the export
    const segs = (json.segments || []).map(s => ({ f: toRel(s.f), team: s.team ?? null, state: s.state || 'controlled' })).sort((a, b) => a.f - b.f);
    const pts = (json.annotations || json.points || []).map(a => ({ ...a, id: a.id || uid(), f: toRel(a.f) }));
    if (layer === 'B') { this.layerB = { segments: segs, points: pts }; }
    else { this._snapshot(); this.segments = segs; this.points = pts; if (json.bookmarks) this.bookmarks = json.bookmarks; }
    this._save();
  }
  clearLayerB() { this.layerB = { segments: [], points: [] }; this._save(); }
  agreement(tolFrames = 12) {
    const n = this.ds.n; let team = 0, state = 0;
    for (let r = 0; r < n; r++) { const a = this.segmentAt(r, false, 'A'), b = this.segmentAt(r, false, 'B'); if (a.team === b.team) team++; if (a.state === b.state) state++; }
    const A = this.items('A').filter(p => p.cat !== 'state'), B = this.items('B').filter(p => p.cat !== 'state'); const usedB = new Set(); let matched = 0;
    for (const a of A) { const j = B.findIndex((b, i) => !usedB.has(i) && b.type === a.type && Math.abs(b.f - a.f) <= tolFrames); if (j >= 0) { usedB.add(j); matched++; } }
    return { team: team / n, state: state / n, events: { matched, onlyA: A.length - matched, onlyB: B.length - matched } };
  }

  // ---------------- exports
  exportJSON(source) {
    return JSON.stringify({ source, start: this.ds.start, end: this.ds.end, fps: this.ds.fps, frames_are_export_relative: true, schema: SCHEMA,
      segments: this.segments, annotations: this.points.map(a => ({ ...a, frame: this.ds.start + a.f, frame_end: this.ds.start + (a.f2 ?? a.f) })), bookmarks: this.bookmarks }, null, 1);
  }
  exportCSV() {
    const rows = ['frame_start,frame_end,duration_s,category,type,team,slot_a,slot_b,x,y,ball_x,ball_y,note'];
    const num = k => this.ds.numOf(k);
    for (const a of this.points) rows.push([this.ds.start + a.f, this.ds.start + (a.f2 ?? a.f), this.durationS(a).toFixed(2), a.cat, a.type, a.team || '',
      a.slots ? num(a.slots[0]) : '', a.slots && a.slots[1] !== undefined ? num(a.slots[1]) : '', a.x !== undefined ? a.x.toFixed(4) : '', a.y !== undefined ? a.y.toFixed(4) : '',
      a.bx !== undefined ? a.bx.toFixed(4) : '', a.by !== undefined ? a.by.toFixed(4) : '', '"' + (a.note || '').replace(/"/g, '""') + '"'].join(','));
    return rows.join('\n');
  }
  // on purpose not the submission format, see rule 1 of the challenge
  exportHumanReading(source, annotator = 'A') {
    const rows = ['frame,team,state,derived_label,annotator'];
    for (let r = 0; r < this.ds.n; r++) { const s = this.segmentAt(r, false, annotator); rows.push(`${this.ds.start + r},${s.team || ''},${s.state === 'clear' ? '' : s.state},${this.derivedLabel(r, annotator)},${annotator}`); }
    return rows.join('\n');
  }
  exportBookmarks() { return 'frame,gt,note\n' + this.bookmarks.map(b => `${this.ds.start + b.f},${b.gt || ''},"${(b.note || '').replace(/"/g, '""')}"`).join('\n'); }
}
