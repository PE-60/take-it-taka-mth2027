// Event lists worked out from the export: kicks, lost ball stretches, jumps and so on.
// All frame indices are relative to the export.
export class EventIndex {
  constructor(ds) {
    this.ds = ds;
    const m = ds.meta, n = ds.n;
    this.kick = new Set(m.events.kick);
    this.lostStarts = []; let inLost = false;
    for (let r = 0; r < n; r++) { const b = ds.ball(r); if (!b && !inLost) this.lostStarts.push(r); inLost = !b; }
    this.jumpFrames = [...new Set([...m.events.bjump.map(e => e[0]), ...m.events.pjump.map(e => e[0])])].sort((a, b) => a - b);
    this.churnFrames = [...new Set([...m.events.vanish.map(e => e[0]), ...m.events.appear.map(e => e[0])])].sort((a, b) => a - b);
    this.gtChanges = []; this.restartFrames = [];
    if (m.gt) for (let r = 1; r < n; r++) {
      if (m.gt[r] !== m.gt[r - 1]) this.gtChanges.push(r);
      if (m.gt[r - 1] === 'D' && (m.gt[r] === 'H' || m.gt[r] === 'A')) this.restartFrames.push(r);
    }
    for (const f of m.flags) { const r = f.f - ds.start; if (r >= 0 && r < n) this.restartFrames.push(r); }
    this.restartFrames = [...new Set(this.restartFrames)].sort((a, b) => a - b);
    this.flags = m.flags.map(f => ({ ...f, r: f.f - ds.start }));
    this.bjump = new Set(m.events.bjump.map(e => e[0]));
    this.pjump = new Set(m.events.pjump.map(e => e[0]));
    // AR flag detections keyed by export-relative frame.
    this.refereeByFrame = new Map();
    this.refereeFrames = [];
    this.refereeSignalFrames = []; // non-none only (for step / candidates)
    if (m.referee && m.referee.events) this.ingestRefereeEvents(m.referee.events, { absolute: false });
  }
  /** Ingest viewer events. absolute=true means e.f / abs_f is a video frame index. */
  ingestRefereeEvents(events, { absolute = false } = {}) {
    if (!events || !events.length) return;
    const n = this.ds.n, start = this.ds.start;
    for (const e of events) {
      let r = absolute ? +(e.abs_f ?? e.f) - start : +e.f;
      if (!Number.isFinite(r) || r < 0 || r >= n) continue;
      if (!this.refereeByFrame.has(r)) this.refereeByFrame.set(r, []);
      this.refereeByFrame.get(r).push(e);
      this.refereeFrames.push(r);
      if (e.signal && e.signal !== 'none') this.refereeSignalFrames.push(r);
    }
    this.refereeFrames = [...new Set(this.refereeFrames)].sort((a, b) => a - b);
    this.refereeSignalFrames = [...new Set(this.refereeSignalFrames)].sort((a, b) => a - b);
  }
  /** Dead-ball intervals from dead_ball_investigation (export/dead_ball.json), on this export's timeline. */
  ingestDeadBall(doc, source) {
    const tl = doc && doc.timelines && doc.timelines[source]; if (!tl) return false;
    const n = this.ds.n, start = this.ds.start;
    const mask = intervals => { const m = new Uint8Array(n); for (const [a, b] of intervals || []) { const r0 = Math.max(0, a - start), r1 = Math.min(n - 1, b - start); if (r1 >= r0) m.fill(1, r0, r1 + 1); } return m; };
    this.dead = { doc, mask: mask(tl.dead_ball), noView: mask(tl.terms && tl.terms.no_view), ballLost: mask(tl.terms && tl.terms.ball_lost), covered: mask(tl.covered) };
    this.deadStarts = []; for (let r = 0; r < n; r++) if (this.dead.mask[r] && (r === 0 || !this.dead.mask[r - 1])) this.deadStarts.push(r);
    return true;
  }
  list(name) {
    switch (name) {
      case 'restart': return this.restartFrames; case 'gt_change': return this.gtChanges; case 'kick': return [...this.kick].sort((a, b) => a - b);
      case 'lost': return this.lostStarts; case 'jump': return this.jumpFrames; case 'churn': return this.churnFrames;
      case 'referee': return this.refereeSignalFrames.length ? this.refereeSignalFrames : this.refereeFrames;
      case 'dead': return this.deadStarts || [];
    }
    return [];
  }
}
export const deriveEvents = ds => new EventIndex(ds);
