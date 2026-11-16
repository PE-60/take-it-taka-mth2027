import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeDataSet } from './fixtures.mjs';
import { slotRuns, orderSlots } from '../../src/slots.js';

test('presence runs per slot', () => {
  const ds = makeDataSet();
  const R = slotRuns(ds);
  assert.deepEqual(R[0], [{ a: 0, b: 299, present: true }]);
  assert.deepEqual(R[4], [{ a: 0, b: 149, present: true }, { a: 150, b: 199, present: false }, { a: 200, b: 299, present: true }]);
});

test('jump marks per slot come from meta', () => {
  const ds = makeDataSet();
  const R = slotRuns(ds);
  assert.deepEqual(R.jumps[1], [210]);
  assert.deepEqual(R.jumps[0], []);
});

test('slots never present in the view are ordered last', () => {
  const ds = makeDataSet();
  const R = slotRuns(ds);
  const order = orderSlots(R, 150, 160);           // slot 4 absent in this view
  assert.equal(order[order.length - 1], 4);
  assert.deepEqual(order.slice(0, 3), [0, 1, 2], 'home first, in slot order');
});
