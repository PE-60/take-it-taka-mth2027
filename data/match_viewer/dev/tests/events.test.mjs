import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeDataSet } from './fixtures.mjs';
import { deriveEvents } from '../../src/events.js';

test('derived event lists', () => {
  const ds = makeDataSet();
  const E = deriveEvents(ds);
  assert.deepEqual([...E.kick], [100]);
  assert.deepEqual(E.lostStarts, [60]);
  assert.deepEqual(E.jumpFrames, [210]);
  assert.deepEqual(E.churnFrames, [150, 200]);
  assert.deepEqual(E.gtChanges, [120, 220, 260]);
  assert.deepEqual(E.restartFrames, [260], 'D -> H at 260');
});

test('event list lookup by name', () => {
  const ds = makeDataSet();
  const E = deriveEvents(ds);
  assert.deepEqual(E.list('restart'), [260]);
  assert.deepEqual(E.list('nonsense'), []);
});
