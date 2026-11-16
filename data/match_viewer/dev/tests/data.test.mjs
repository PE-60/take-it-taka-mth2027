import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeDataSet } from './fixtures.mjs';

test('DataSet exposes frames, ball, slots and sides', () => {
  const ds = makeDataSet();
  assert.equal(ds.n, 300);
  assert.equal(ds.ball(0).x.toFixed(2), '0.30');
  assert.equal(ds.ball(65), null, 'ball lost 60..69');
  assert.equal(ds.players(10).length, 6);
  assert.equal(ds.players(160).length, 5, 'slot 4 vanished');
  assert.equal(ds.sideOf(0), 'H'); assert.equal(ds.sideOf(3), 'A');
});

test('projection and inverse round-trip through the homography', () => {
  const ds = makeDataSet();
  const H = ds.H(0);
  const [px, py] = ds.proj(H, 0.5, 0.25);
  assert.equal(px, 600); assert.equal(py, 250);
  const back = ds.pitchFromPx(0, px, py);
  assert.ok(Math.abs(back.x - 0.5) < 1e-9 && Math.abs(back.y - 0.25) < 1e-9);
});

test('metres uses the pitch size', () => {
  const ds = makeDataSet();
  assert.equal(ds.metres({ x: 0, y: 0 }, { x: 1, y: 0 }), 105.3);
  assert.equal(ds.metres({ x: 0, y: 0 }, { x: 0, y: 1 }), 68);
});

test('attack direction flips in the second half', () => {
  const ds = makeDataSet();
  assert.equal(ds.attackDir(100), 1);
  assert.equal(ds.attackDir(7000), -1);
});
