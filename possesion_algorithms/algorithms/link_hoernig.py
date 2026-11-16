"""Possession after Link & Hoernig (2017), as the match viewer used to compute its L&H row.

A player gains the ball when he is the nearest one and within t_p of it. He keeps it while the ball
1 s ahead stays within t_a of where he is 1 s ahead; a kick (ball acceleration >= 4 m/s²) re-opens
the gain test, so the nearest other player within t_p takes over. The team of the last holder is
sticky: through loose balls and frames without a tracked ball, possession stays with it.
"""

import numpy as np

from possesion_algorithms.base import (BROADCAST, TACTICAL, TACTICAL_THEN_BROADCAST, PossessionAlgorithm,
                                       Tracking)


class LinkHoernig(PossessionAlgorithm):
    name = "lh"
    description = "Link & Hoernig: gain within t_p, keep while the ball 1 s ahead is within t_a; whole video, tactical"
    dead_ball = False
    source = TACTICAL
    params = {"t_p": 1.0, "t_a": 2.5, "horizon_s": 1.0}

    def label(self, track: Tracking) -> np.ndarray:
        n, t_p, t_a = track.n, self.params["t_p"], self.params["t_a"]
        out = np.full(n, "", dtype="<U1")
        if n == 0 or track.n_slots == 0:
            return out
        d = track.ball_distances()
        nearest = np.argmin(d, axis=1)
        near_d = d[np.arange(n), nearest]
        seen = np.isfinite(track.ball[:, 0])
        present = np.isfinite(track.xy[:, :, 0])
        h = int(self.params["horizon_s"] * track.fps)
        holder, last_team = -1, ""
        for r in range(n):
            if not seen[r]:
                holder = -1
                out[r] = last_team
                continue
            k, dk = nearest[r], near_d[r]
            if holder >= 0 and present[r, holder]:
                r2 = min(n - 1, r + h)
                ahead = d[r2, holder] if np.isfinite(d[r2, holder]) else d[r, holder]
                if ahead > t_a or (track.kick[r] and dk <= t_p and k != holder):
                    holder = -1
            else:
                holder = -1
            if holder < 0 and dk <= t_p:
                holder = k
            if holder >= 0:
                last_team = track.team[holder]
            out[r] = last_team
        return out


class LinkHoernigDeadBall(LinkHoernig):
    name = "lh_dead"
    description = "lh on each alive interval of dead_ball.json on its own, D on dead balls; tactical"
    dead_ball = True


class LinkHoernigDeadBallBroadcast(LinkHoernigDeadBall):
    name = "lh_dead_b"
    description = "lh_dead on the broadcast tracking, synced onto the tactical timeline"
    source = BROADCAST


class LinkHoernigDeadBallFallback(LinkHoernigDeadBall):
    name = "lh_dead_tb"
    description = "lh_dead on the tactical tracking, falling back to the synced broadcast where tactical lost the ball"
    source = TACTICAL_THEN_BROADCAST
