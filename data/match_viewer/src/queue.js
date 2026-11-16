// A list of candidate frames, each with a status, so you can work through them and come back later.
export const STATUSES = ['todo', 'done', 'skipped', 'flagged'];

export class CandidateQueue {
  constructor(frames, saved = null) {
    this.frames = [...frames].sort((a, b) => a - b);
    this.status = new Map();
    if (saved) for (const [f, s] of Object.entries(saved)) if (this.frames.includes(+f) && STATUSES.includes(s) && s !== 'todo') this.status.set(+f, s);
  }
  statusOf(f) { return this.status.get(f) || 'todo'; }
  setStatus(f, s) { if (!STATUSES.includes(s)) throw new Error('bad status ' + s); if (s === 'todo') this.status.delete(f); else this.status.set(f, s); }
  handled(f) { return this.statusOf(f) !== 'todo'; }
  /** next candidate after/before `cur` (exclusive); skips handled unless includeHandled; wraps. null if none. */
  next(cur, dir = 1, { includeHandled = false } = {}) {
    const ok = f => includeHandled || !this.handled(f);
    const L = this.frames.filter(ok); if (!L.length) return null;
    if (dir > 0) { const t = L.find(f => f > cur); return t === undefined ? L[0] : t; }
    const t = [...L].reverse().find(f => f < cur); return t === undefined ? L[L.length - 1] : t;
  }
  markAndAdvance(f, s) { this.setStatus(f, s); return this.next(f, 1); }
  progress() {
    const p = { done: 0, skipped: 0, flagged: 0, total: this.frames.length, remaining: 0 };
    for (const f of this.frames) { const s = this.statusOf(f); if (s === 'todo') p.remaining++; else p[s]++; }
    return p;
  }
  serialize() { return Object.fromEntries(this.status); }
}
