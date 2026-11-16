import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeDataSet, makeDataSetWithBallq } from './fixtures.mjs';

test('without ballq.bin the dataset says so and returns nulls', () => {
  const ds = makeDataSet();
  assert.equal(ds.hasBallq(), false);
  assert.equal(ds.ballq(10), null); assert.equal(ds.lhPost(10), null);
  assert.ok(Number.isNaN(ds.ballMoved(10)));
});

test('filtered ball and both heuristics are read per frame', () => {
  const ds = makeDataSetWithBallq();
  assert.equal(ds.hasBallq(), true);
  const q = ds.ballq(10);
  assert.ok(Math.abs(q.x - ds.ball(10).x) < 1e-6 && q.tau > 0.9 && q.flags === 0);
  const lost = ds.ballq(65);
  assert.equal(ds.ball(65), null, 'detector lost');
  assert.ok(lost && lost.flags === 1 && lost.tau === 0, 'filter still has a ball, flagged lost, detector not trusted');
  assert.equal(ds.lhRaw(115), 'A'); assert.equal(ds.lhPost(115), 'H'); assert.equal(ds.gt(115), 'H');
});

test('ballMoved is the metre distance between detector and filtered mean', () => {
  const ds = makeDataSetWithBallq();
  assert.ok(ds.ballMoved(10) < 0.01);
});

test('team names come from the export, with a fallback', () => {
  assert.equal(makeDataSet().teamName('H'), 'home');
  assert.equal(makeDataSet({ team_names: { H: 'Alpha FC', A: 'Beta' } }).teamName('A'), 'Beta');
});

test('lhAgreement counts frames the filtered heuristic fixes and breaks', async () => {
  // app.js touches the DOM at import time, so the pure function is copied here in the same form
  const lhAgreement = (ds, lhRaw) => {
    let n = 0, raw = 0, post = 0, onlyPost = 0, onlyRaw = 0;
    for (let r = 0; r < ds.n; r++) { const g = ds.gt(r); if (g !== 'H' && g !== 'A') continue; n++; const a = lhRaw[r] === g, b = ds.lhPost(r) === g; raw += a; post += b; onlyPost += b && !a; onlyRaw += a && !b; }
    return { n, raw: raw / n, post: post / n, onlyPost, onlyRaw };
  };
  const ds = makeDataSetWithBallq();
  const ag = lhAgreement(ds, ds.meta.ballq.lh_raw.split(''));
  assert.equal(ag.n, 260);
  assert.equal(ag.onlyPost, 10, 'frames 110..119: raw says A, filtered and GT say H');
  assert.equal(ag.onlyRaw, 0);
  assert.ok(ag.post > ag.raw);
});

test('grid cells decode from uint16 triples and map to pitch boxes', () => {
  const ds = makeDataSetWithBallq();
  ds.meta.ballq.grid = { cell_m: 0.25, nrows: 272, ncols: 422 };
  const cells = ds.decodeGrid(new Uint16Array([10, 20, 65535, 11, 20, 32768]));
  assert.equal(cells.n, 2); assert.ok(Math.abs(cells.p[0] - 1) < 1e-6 && Math.abs(cells.p[1] - 0.5) < 1e-3 && cells.max > 0.99);
  const [x0, y0, x1, y1] = ds.cellBox(10, 20);
  assert.ok(Math.abs((x1 - x0) * ds.pitch[0] - 0.25) < 1e-6 && Math.abs((y1 - y0) * ds.pitch[1] - 0.25) < 1e-6);
  assert.ok(Math.abs(x0 * ds.pitch[0] - 5) < 1e-6 && Math.abs(y0 * ds.pitch[1] - 2.5) < 1e-6);
  assert.equal(ds.hasGrid(), false, 'no index loaded yet');
});

test('our dead ball overrides the filtered L&H in the submission label', () => {
  const ds = makeDataSetWithBallq();
  assert.equal(ds.hasDeadOurs(), false); assert.equal(ds.submission(10), ds.lhPost(10), 'without a dead-ball file the submission is the filtered L&H');
  ds.deadMask = Uint8Array.from({ length: ds.n }, (_, r) => (r >= 50 && r < 60 ? 1 : 0));   // what app.js hands over from export/dead_ball.json
  assert.equal(ds.hasDeadOurs(), true);
  assert.equal(ds.deadOurs(55), true); assert.equal(ds.submission(55), 'D');
  assert.equal(ds.submission(10), ds.lhPost(10));
  assert.equal(ds.submission(230), null, 'X from L&H stays an abstention');
});

test('the submission row absorbs a short blip between the same team, like the CSV does', () => {
  const ds = makeDataSetWithBallq();
  // filtered L&H: H everywhere except a 0.4 s blip of A at 40..49, which the fixture ds.lhPost reads from meta
  ds.meta.ballq.lh_post = 'H'.repeat(40) + 'A'.repeat(10) + 'H'.repeat(ds.n - 50);
  ds.deadMask = new Uint8Array(ds.n);
  assert.equal(ds.submission(45), 'H');
  ds.meta.ballq.lh_post = 'H'.repeat(40) + 'A'.repeat(ds.n - 40); ds.resetSubmission();
  assert.equal(ds.submission(45), 'A', 'a real change stays');
});
