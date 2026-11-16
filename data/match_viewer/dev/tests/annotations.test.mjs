import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeDataSet, MemoryStorage } from './fixtures.mjs';
import { AnnotationStore, SCHEMA, TOOL } from '../../src/annotations.js';

const mk = () => { const ds = makeDataSet(); return { ds, st: new AnnotationStore(new MemoryStorage(), 'k', ds) }; };

test('S4 one segment model: team and state, inheritance, derived label', () => {
  const { st } = mk();
  st.setSegment(10, { team: 'H' });
  assert.deepEqual(st.segmentAt(50), { team: 'H', state: 'controlled' });
  st.setSegment(30, { state: 'flight' });
  assert.deepEqual(st.segmentAt(35), { team: 'H', state: 'flight' }, 'state change keeps the team');
  st.setSegment(40, { state: 'dead' });
  assert.equal(st.derivedLabel(45), 'D');
  st.setSegment(60, { team: 'A' });
  assert.deepEqual(st.segmentAt(61), { team: 'A', state: 'controlled' }, 'new team after dead resets to controlled');
  st.setSegment(80, null);
  assert.equal(st.derivedLabel(85), 'X');
  assert.equal(st.derivedLabel(5), 'X');
});

test('S4 setting team then state on the same frame keeps the team', () => {
  const { st } = mk();
  st.setSegment(20, { team: 'H' }); st.setSegment(20, { state: 'flight' });
  assert.deepEqual(st.segmentAt(25), { team: 'H', state: 'flight' });
  st.setSegment(20, { team: 'A' });
  assert.deepEqual(st.segmentAt(25), { team: 'A', state: 'flight' }, 'team change on the same frame keeps the alive state');
});

test('S4 migration of legacy labels into segments', () => {
  const ds = makeDataSet(); const mem = new MemoryStorage();
  mem.set('k', JSON.stringify({ labels: { 10: 'H', 50: 'D', 70: 'A' }, bookmarks: [], ann: [] }));
  const st = new AnnotationStore(mem, 'k', ds);
  assert.equal(st.derivedLabel(20), 'H'); assert.equal(st.derivedLabel(55), 'D'); assert.equal(st.derivedLabel(90), 'A');
  assert.equal(st.segmentAt(20).state, 'controlled');
});

test('S2 intervals open and close with the same type; swap; discard; duration', () => {
  const { st } = mk();
  st.beginInterval(120, { cat: 'data', type: 'homography_off' });
  assert.ok(st.pending && st.pending.type === 'homography_off');
  st.endInterval(100);                              // closed before start -> swapped
  const a = st.items('A').find(x => x.type === 'homography_off');
  assert.equal(a.f, 100); assert.equal(a.f2, 120);
  assert.equal(st.durationS(a), 21 / 25);
  st.beginInterval(5, { cat: 'data', type: 'gt_doubt' });
  st.discardPending();
  assert.equal(st.pending, null);
  assert.equal(st.items('A').filter(x => x.type === 'gt_doubt').length, 0);
});

test('S2 interval covers frames for the pitch/time raster', () => {
  const { st } = mk();
  st.beginInterval(10, { cat: 'data', type: 'ball_pos', x: 0.5, y: 0.5 }); st.endInterval(20);
  assert.equal(st.at(15).length, 1); assert.equal(st.at(21).length, 0);
});

test('S5 nudge moves a pending position by metres', () => {
  const { st, ds } = mk();
  st.beginInterval(10, { cat: 'data', type: 'ball_pos', x: 0.5, y: 0.5 });
  st.nudgePending(0.1, 0);                          // 10 cm along x
  assert.ok(Math.abs(st.pending.x - (0.5 + 0.1 / ds.pitch[0])) < 1e-9);
  st.nudgePending(0, -0.5);
  assert.ok(Math.abs(st.pending.y - (0.5 - 0.5 / ds.pitch[1])) < 1e-9);
});

test('S9 undo reverts the last change, up to 50', () => {
  const { st } = mk();
  st.addPoint(10, { cat: 'event', type: 'pass' });
  st.setSegment(10, { team: 'H' });
  assert.equal(st.undo(), true);
  assert.equal(st.derivedLabel(10), 'X');
  assert.equal(st.undo(), true);
  assert.equal(st.items('A').length, 0);
  assert.equal(st.undo(), false);
  for (let i = 0; i < 60; i++) st.addPoint(i, { cat: 'event', type: 'pass' });
  let n = 0; while (st.undo()) n++;
  assert.equal(n, 50);
});

test('S3 layer B import and agreement', () => {
  const { st } = mk();
  st.setSegment(0, { team: 'H' }); st.setSegment(150, { team: 'A' });
  st.addPoint(100, { cat: 'event', type: 'pass' }); st.addPoint(200, { cat: 'event', type: 'tackle' });
  const other = { segments: [{ f: 0, team: 'H', state: 'controlled' }, { f: 80, team: 'H', state: 'flight' }, { f: 100, team: 'A', state: 'flight' }, { f: 150, team: 'A', state: 'controlled' }],
    annotations: [{ f: 104, cat: 'event', type: 'pass' }, { f: 250, cat: 'event', type: 'shot' }] };
  st.importLayer(other, 'B');
  assert.equal(st.items('B').length, 2);
  const ag = st.agreement(12);
  assert.ok(Math.abs(ag.team - 250 / 300) < 1e-9, 'frames 100..149 disagree on team');
  assert.ok(Math.abs(ag.state - 230 / 300) < 1e-9, 'frames 80..149 disagree on state');
  assert.deepEqual(ag.events, { matched: 1, onlyA: 1, onlyB: 1 });
});

test('S10 human reading export is not in submission format', () => {
  const { st } = mk();
  st.setSegment(0, { team: 'H' }); st.setSegment(10, { state: 'dead' });
  const csv = st.exportHumanReading('tactical');
  const lines = csv.split('\n');
  assert.equal(lines[0], 'frame,team,state,derived_label,annotator');
  assert.equal(lines[1], '1000,H,controlled,H,A');
  assert.equal(lines[11], '1010,,dead,D,A');
  assert.equal(lines.length, 301);
});

test('schema: every tool has a unique key and a category', () => {
  const keys = Object.keys(TOOL);
  assert.equal(new Set(keys).size, keys.length);
  for (const k of keys) assert.ok(SCHEMA[TOOL[k].cat]);
});

test('persistence: store round-trips through the adapter', () => {
  const ds = makeDataSet(); const mem = new MemoryStorage();
  const st = new AnnotationStore(mem, 'k', ds);
  st.addPoint(3, { cat: 'event', type: 'shot' }); st.setSegment(3, { team: 'A' }); st.addBookmark(3, 'hello');
  const st2 = new AnnotationStore(mem, 'k', ds);
  assert.equal(st2.items('A').length, 1); assert.equal(st2.derivedLabel(4), 'A'); assert.equal(st2.bookmarks[0].note, 'hello');
});
