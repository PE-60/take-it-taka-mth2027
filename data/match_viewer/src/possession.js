// Individual ball possession after Link and Hoernig (2017), the same rule as ballq/possession.py:
//   start  the ball is inside tp of the nearest player and either a kick happened within kickWindow
//          (a ball action: local maximum of the xy acceleration >= 4 m/s2) or the player controls it
//          (ball and player move together, relative speed <= ctrlRel, and the ball predicted a second
//          ahead is still within ta). A ball rolling or flying past fails both and changes nothing.
//   end    the predicted ball stays outside ta for endHold seconds (at once after a kick), or another
//          player starts a possession.
// naive = nearest player within tp, kept as the comparison the paper itself made.
// The team stays with the last holder while nobody holds.
export function computePossession(ds, kickSet, tp, ta = 2.5, horizonS = 1.0, opts = {}) {
  const { kickWindowS = 0.2, ctrlRel = 2.0, endHoldS = 1.0 } = opts;
  const n = ds.n, S = ds.S, fps = ds.fps;
  const naive = new Array(n).fill(''), lh = new Array(n).fill('');
  const naivePid = new Int16Array(n).fill(-1), lhPid = new Int16Array(n).fill(-1);
  const side = []; for (let k = 0; k < S; k++) side.push(ds.sideOf(k));
  const w = Math.round(kickWindowS * fps), endHold = Math.round(endHoldS * fps);
  let holder = -1, lastTeam = '', far = 0;
  const dist = new Float32Array(S);
  const kickNear = r => { for (let k = r - w; k <= r + w; k++) if (kickSet.has(k)) return true; return false; };
  // velocity in m/s from the frame before (the frame after on the first frame), null when missing
  const vel = (get, r) => { for (const [a, b] of [[r - 1, r], [r, r + 1]]) { if (a < 0 || b >= n) continue; const p = get(a), q = get(b); if (p && q) return [(q.x - p.x) * ds.pitch[0] * fps, (q.y - p.y) * ds.pitch[1] * fps]; } return null; };
  for (let r = 0; r < n; r++) {
    const b = ds.ball(r); if (!b) { holder = -1; far = 0; lh[r] = lastTeam; continue; }
    let best = -1, bd = Infinity;
    for (let k = 0; k < S; k++) { const p = ds.slot(r, k); if (!p) { dist[k] = Infinity; continue; } const d = ds.metres(b, p); dist[k] = d; if (d < bd) { bd = d; best = k; } }
    if (best >= 0 && bd <= tp) { naive[r] = side[best]; naivePid[r] = best; }
    const vb = vel(i => ds.ball(i), r);
    const ahead = k => { if (!vb) return Infinity; const p = ds.slot(r, k); if (!p) return Infinity; return Math.hypot(b.x * ds.pitch[0] + vb[0] * horizonS - p.x * ds.pitch[0], b.y * ds.pitch[1] + vb[1] * horizonS - p.y * ds.pitch[1]); };
    const gains = k => {
      if (k < 0 || dist[k] > tp) return false;
      if (kickNear(r)) return true;
      const vp = vel(i => ds.slot(i, k), r); if (!vp || !vb) return false;
      return Math.hypot(vb[0] - vp[0], vb[1] - vp[1]) <= ctrlRel && ahead(k) <= ta;
    };
    if (holder >= 0 && ds.slot(r, holder)) {
      far = ahead(holder) > ta ? far + 1 : 0;
      const taken = best !== holder && gains(best);
      if (far >= (kickNear(r) ? 1 : endHold) || taken) { holder = -1; far = 0; }
    } else { holder = -1; far = 0; }
    if (holder < 0 && gains(best)) holder = best;
    if (holder >= 0) { lastTeam = side[holder]; lhPid[r] = holder; }
    lh[r] = lastTeam;                                   // the team stays through flight, duels and lost tracking
  }
  return { naive, lh, naivePid, lhPid, tp };
}
