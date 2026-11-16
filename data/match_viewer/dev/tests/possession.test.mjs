import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeDataSet } from './fixtures.mjs';
import { computePossession } from '../../src/possession.js';

test('naive follows the nearest player inside the radius, empty in flight', () => {
  const ds = makeDataSet();
  const P = computePossession(ds, new Set([100]), 1.0);
  assert.equal(P.naive[50], 'H');
  assert.equal(P.naive[115], '', 'mid-flight nobody within 1 m');
  assert.equal(P.naive[200], 'A');
});

test('L&H keeps the kicker through the flight, then hands over', () => {
  const ds = makeDataSet();
  const P = computePossession(ds, new Set([100]), 1.0);
  assert.equal(P.lh[50], 'H');
  assert.equal(P.lh[105], 'H', 'team keeps credit through the flight'); assert.equal(P.lhPid[105], -1, 'but no individual holder in flight');
  assert.equal(P.lh[200], 'A');
  assert.equal(P.lhPid[200], 3);
});

test('lost ball clears naive; the L&H team is sticky (challenge rule: loose ball stays with the last team)', () => {
  const ds = makeDataSet();
  const P = computePossession(ds, new Set(), 1.0);
  assert.equal(P.naive[65], ''); assert.equal(P.lh[65], 'H'); assert.equal(P.lhPid[65], -1);
});

test('a pass rolling past a defender does not flip possession; a trap starts the receiver (Link and Hoernig with kick detection)', () => {
  // the fixture ball flies 30 m from frame 100 to 130; place an away defender 0.6 m off the line halfway
  const ds = makeDataSet();
  const PITCH = ds.pitch, B = ds.meta.block;
  for (let r = 0; r < ds.n; r++) { const o = r * B + 9 + 6 + 4 * 4; ds.D[o] = 0.45; ds.D[o + 1] = 0.5 + 0.6 / PITCH[1]; }   // slot 4 (away) parked on the pass line
  const P = computePossession(ds, new Set([100]), 1.0);
  const near = []; for (let r = 100; r < 130; r++) { const b = ds.ball(r), p = ds.slot(r, 4); if (b && p && ds.metres(b, p) <= 1.0) near.push(r); }
  assert.ok(near.length > 0, 'the ball does pass inside the defender\'s metre');
  for (const r of near) assert.equal(P.lh[r], 'H', `frame ${r}: the pass does not change the team`);
  assert.ok(!Array.from(P.lhPid).includes(4), 'the defender never becomes the holder');
  assert.equal(P.lh[200], 'A', 'the receiver who carries the ball does');
});
