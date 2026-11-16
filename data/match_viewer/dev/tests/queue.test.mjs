import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CandidateQueue } from '../../src/queue.js';

test('next/prev skip handled candidates and wrap', () => {
  const q = new CandidateQueue([10, 20, 30, 40]);
  assert.equal(q.next(10, 1), 20);
  q.setStatus(20, 'done');
  assert.equal(q.next(10, 1), 30, 'skips done');
  assert.equal(q.next(10, 1, { includeHandled: true }), 20);
  assert.equal(q.next(40, 1), 10, 'wraps');
  assert.equal(q.next(10, -1), 40);
  assert.equal(q.next(25, 1), 30, 'from a frame between candidates');
});

test('progress counts and remaining', () => {
  const q = new CandidateQueue([1, 2, 3, 4, 5]);
  q.setStatus(1, 'done'); q.setStatus(2, 'skipped'); q.setStatus(3, 'flagged');
  assert.deepEqual(q.progress(), { done: 1, skipped: 1, flagged: 1, total: 5, remaining: 2 });
  q.setStatus(1, 'todo');
  assert.equal(q.progress().remaining, 3);
});

test('mark and advance', () => {
  const q = new CandidateQueue([1, 2, 3]);
  assert.equal(q.markAndAdvance(1, 'done'), 2);
  assert.equal(q.markAndAdvance(2, 'done'), 3);
  assert.equal(q.markAndAdvance(3, 'done'), null, 'nothing left');
});

test('serialise round trip keeps statuses for frames still in the list', () => {
  const q = new CandidateQueue([1, 2, 3]);
  q.setStatus(2, 'flagged');
  const q2 = new CandidateQueue([2, 3, 9], q.serialize());
  assert.equal(q2.statusOf(2), 'flagged');
  assert.equal(q2.statusOf(9), 'todo');
});
