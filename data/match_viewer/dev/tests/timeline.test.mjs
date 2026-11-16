import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeDataSet } from './fixtures.mjs';
import { deriveEvents } from '../../src/events.js';
import { computePossession } from '../../src/possession.js';
import { rasterLabel, rasterMask, rasterIntensity, rasterInterval, aggregateBands } from '../../src/timeline.js';

const BG = [34, 34, 34];
const cmap = { H: [235, 70, 70], A: [50, 170, 235], D: [189, 189, 189] };

test('label raster takes the majority label per pixel', () => {
  const ds = makeDataSet();
  const px = rasterLabel(r => ds.meta.gt[r], 0, 299, 30, cmap, BG);   // 10 frames per pixel
  assert.deepEqual([...px.slice(0, 3)], cmap.H);
  assert.deepEqual([...px.slice(15 * 3, 15 * 3 + 3)], cmap.A);
  assert.deepEqual([...px.slice(23 * 3, 23 * 3 + 3)], cmap.D);
});

test('mask raster lights a pixel if any frame in it matches', () => {
  const px = rasterMask(r => r === 65, 0, 299, 30, [255, 0, 0], BG);
  assert.deepEqual([...px.slice(6 * 3, 6 * 3 + 3)], [255, 0, 0]);
  assert.deepEqual([...px.slice(7 * 3, 7 * 3 + 3)], BG);
});

test('intensity raster scales to vmax and flags bad values', () => {
  const px = rasterIntensity(r => (r < 10 ? 15 : r < 20 ? 50 : NaN), 0, 29, 3, 30, [255, 255, 255], BG, { bad: 40, badColor: [240, 60, 60] });
  assert.deepEqual([...px.slice(0, 3)].map(Math.round), [144, 144, 144]);
  assert.deepEqual([...px.slice(3, 6)], [240, 60, 60]);
  assert.deepEqual([...px.slice(6, 9)], BG);
});

test('interval raster paints every pixel the interval touches', () => {
  const px = rasterInterval([{ f: 100, f2: 129, color: [1, 2, 3] }], 0, 299, 30, BG);
  assert.deepEqual([...px.slice(10 * 3, 10 * 3 + 3)], [1, 2, 3]);
  assert.deepEqual([...px.slice(12 * 3, 12 * 3 + 3)], [1, 2, 3]);
  assert.deepEqual([...px.slice(13 * 3, 13 * 3 + 3)], BG);
});

test('aggregate bands per bin: restarts, lost fraction, team share', () => {
  const ds = makeDataSet();
  const E = deriveEvents(ds);
  const P = computePossession(ds, E.kick, 1.0);
  const B = aggregateBands(ds, E, P, 100);      // 3 bins of 100 frames
  assert.equal(B.length, 3);
  assert.equal(B[0].lostFrac, 0.10);
  assert.equal(B[2].restarts, 1);
  assert.ok(B[0].shareH > 0.9 && B[0].shareA === 0);
  assert.ok(B[2].shareA > B[2].shareH);
  assert.equal(B[1].kicks, 1, 'the kick at frame 100 lands in bin 1');
  assert.equal(B[1].countBad, 0.5, 'slot 4 is gone for 150..199, half of bin 1');
  assert.equal(B[0].homBad, 0, 'homography fine everywhere in the fixture');
});
