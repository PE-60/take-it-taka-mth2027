// For each track slot: when it was present, when it was gone, and where it jumped.
export function slotRuns(ds) {
  const out = []; out.jumps = [];
  for (let k = 0; k < ds.S; k++) {
    const runs = []; let cur = null;
    for (let r = 0; r < ds.n; r++) {
      const p = ds.present(r, k);
      if (!cur || cur.present !== p) { cur = { a: r, b: r, present: p }; runs.push(cur); } else cur.b = r;
    }
    out.push(runs); out.jumps.push([]);
  }
  for (const [r, k] of ds.meta.events.pjump) out.jumps[k].push(r);
  return out;
}
/** slot order for a view: home first then away, in slot order; slots with no presence in [r0, r1] last */
export function orderSlots(runs, r0, r1) {
  const S = runs.length, seen = k => runs[k].some(x => x.present && x.b >= r0 && x.a <= r1);
  const idx = [...Array(S).keys()];
  const team = runs.team || (k => (k < S / 2 ? 'H' : 'A'));
  const key = k => (seen(k) ? 0 : 1) * 1000 + (team(k) === 'H' ? 0 : 500) + k;
  return idx.sort((a, b) => key(a) - key(b));
}
