// End-to-end: drives the served viewer through S1..S12.
// Usage: node match_viewer/dev/tests/e2e.mjs http://127.0.0.1:8780/match_viewer [export/tactical_gt]
import { createRequire } from 'node:module';
const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_PATH || 'playwright');
const base = process.argv[2] || 'http://127.0.0.1:8780/match_viewer'; const exportDir = process.argv[3] || 'export/tactical_gt';
const results = []; const check = (name, ok, info = '') => { results.push([ok, name, info]); };

const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
const errs = []; p.on('pageerror', e => errs.push(e.stack));
await p.goto(base + '/match_viewer.html?d=' + exportDir, { waitUntil: 'networkidle' }); await p.waitForTimeout(2500);
const pv = (f, a) => p.evaluate(f, a);
check('loads data', await pv(() => !!window.__pv.ds && window.__pv.ds.n > 0));

// S4 segments
await pv(() => window.__pv.seek(20)); await p.keyboard.press('h'); await p.keyboard.press('2');
await pv(() => window.__pv.seek(60)); await p.keyboard.press('d');
check('S4 segment team+state, derived label', await pv(() => window.__pv.store.segmentAt(30).team === 'H' && window.__pv.store.segmentAt(30).state === 'flight' && window.__pv.store.derivedLabel(65) === 'D'));

// S2 intervals: open U at 70, close at 90
await pv(() => window.__pv.seek(70)); await p.keyboard.press('u'); check('S2 pending interval shown', await pv(() => !!window.__pv.store.pending));
await pv(() => window.__pv.seek(90)); await p.keyboard.press('u');
check('S2 interval closed with f2', await pv(() => { const a = window.__pv.store.items('A').find(x => x.type === 'homography_off'); return a && a.f === 70 && a.f2 === 90; }));

// S5 ball position: N places at tracked ball, nudge, Enter closes
await pv(() => window.__pv.seek(95)); await p.keyboard.press('n');
const before = await pv(() => window.__pv.store.pending && window.__pv.store.pending.x);
await p.keyboard.press('ArrowRight'); await p.keyboard.press('ArrowRight');
const after = await pv(() => window.__pv.store.pending.x);
check('S5 nudge moved 20 cm', Math.abs((after - before) * 105.3 - 0.2) < 1e-6, `${((after - before) * 105.3).toFixed(3)} m`);
check('S5 video paused by tool', await pv(() => document.querySelector('#v').paused));
await p.keyboard.press('Enter'); check('S5 Enter closes', await pv(() => !window.__pv.store.pending && window.__pv.store.items('A').some(a => a.type === 'ball_pos')));

// M + click + team
const c = await pv(() => { const ov = document.querySelector('#ov'), r = ov.getBoundingClientRect(); const H = window.__pv.ds.H(window.__pv.cur); const q = window.__pv.ds.proj(H, 0.5, 0.45); return [r.left + q[0] * r.width / ov.width, r.top + q[1] * r.height / ov.height]; });
await p.keyboard.press('m'); await p.mouse.click(c[0], c[1]); await p.keyboard.press('a'); await p.keyboard.press('Enter');
check('S2 player_missing with team via click + key', await pv(() => window.__pv.store.items('A').some(a => a.type === 'player_missing' && a.team === 'A' && a.x !== undefined)));

// events
await p.keyboard.press('p'); check('event point added', await pv(() => window.__pv.store.items('A').some(a => a.type === 'pass')));

// S9 undo
const n0 = await pv(() => window.__pv.store.items('A').length); await p.keyboard.press('Control+z');
check('S9 undo removed last', await pv(n => window.__pv.store.items('A').length === n - 1, n0));

// S1 queue
await pv(() => window.__pv.setCandType('kick')); await p.waitForTimeout(200);
const first = await pv(() => window.__pv.queue.frames[0]); await pv(f => window.__pv.seek(f), first);
await p.keyboard.press('Enter');
check('S1 Enter marks done and persists', await pv(([f, d]) => window.__pv.queue.statusOf(f) === 'done' && JSON.parse(localStorage.getItem('pv2:' + d)).queues.kick[f] === 'done', [first, exportDir]));
check('S1 progress text', (await p.textContent('#candN')).includes('done 1'));

// S3 layer B + agreement
await pv(() => { window.__pv.store.importLayer({ segments: [{ f: 0, team: 'H', state: 'controlled' }], annotations: [{ f: 96, cat: 'event', type: 'pass' }] }, 'B'); });
await p.keyboard.press('Escape'); await p.waitForTimeout(100);
const ag = await pv(() => window.__pv.store.agreement(12));
check('S3 agreement computed', ag && typeof ag.team === 'number' && ag.events.onlyB >= 0, JSON.stringify(ag));

// S7 slots group toggle via label click
const slotsBefore = await pv(() => [...document.querySelectorAll('#rowlabels div')].filter(d => d.textContent.startsWith('#')).length);
await p.click('#rowlabels div.grp:has-text("slots")'); await p.waitForTimeout(200);
const slotsAfter = await pv(() => [...document.querySelectorAll('#rowlabels div')].filter(d => d.textContent.startsWith('#')).length);
check('S7 slots group opens', slotsBefore === 0 && slotsAfter === 46, `${slotsBefore} -> ${slotsAfter}`);
await p.click('#rowlabels div:has-text("#3")'); check('S7 clicking slot row pins', await pv(() => window.__pv.pinned.length === 1));

// S8 overview bands clickable: click near the top of the timeline canvas at 50%
const before8 = await pv(() => window.__pv.cur); await pv(() => { const tb = document.querySelector('#tlbody'); if (tb) tb.scrollTop = 0; });
const tl = await (await p.$('#tlc')).boundingBox(); await p.mouse.click(tl.x + tl.width * 0.5, tl.y + 4);
check('S8 overview click seeks', await pv(b => window.__pv.cur !== b && window.__pv.cur > window.__pv.ds.n * 0.3 && window.__pv.cur < window.__pv.ds.n * 0.7, before8), String(await pv(() => window.__pv.cur)));

// S6 loop
await p.keyboard.press('l'); await p.waitForTimeout(300); check('S6 loop active', (await p.textContent('#loopInd')).startsWith('loop')); await p.keyboard.press('l');

// S9 help overlay, S10 export name
await p.keyboard.press('?'); check('S9 help overlay', await pv(() => document.querySelector('#help').style.display === 'block')); await p.keyboard.press('Escape');
const hr = await pv(() => window.__pv.store.exportHumanReading('tactical').split('\n')[0]);
check('S10 human reading header', hr === 'frame,team,state,derived_label,annotator', hr);

// S11 overview: GT first, toggle a band
check('S11 GT is first in the overview', await pv(() => window.__pv.overviewOn()[0] === 'GT'));
await p.click('#bOverview'); await p.click('#ovpop label:has-text("kicks") input'); await p.waitForTimeout(150);
check('S11 overview toggle persists', await pv(d => window.__pv.overviewOn().includes('kicks') && JSON.parse(localStorage.getItem('pv2:' + d)).overview.includes('kicks'), exportDir));
await p.keyboard.press('Escape');
// rows filter + legend coupling
await p.click('#bRows'); await p.click('#rowpop .presets button:has-text("minimal")'); await p.waitForTimeout(200);
check('legend hides entries of hidden rows', await pv(() => ![...document.querySelectorAll('#legend span')].some(s => s.textContent.trim() === 'kick')));
// S12 filtered ball: rows, candidates, header numbers (only when the export carries ballq)
const hasQ = await pv(() => window.__pv.ds.hasBallq());
if (hasQ) {
  if (!(await p.isVisible('#rowpop'))) await p.click('#bRows'); await p.click('#rowpop .presets button:has-text("ball filter")'); await p.waitForTimeout(200); await p.keyboard.press('Escape');
  check('S12 filtered rows present', await pv(() => ['L&H FILTERED', 'DISAGREE', 'BALL MOVED', 'TRUST'].every(n => window.__pv.rows().some(r => r.name === n))));
  check('S12 header shows raw and filtered agreement', (await p.textContent('#agree')).includes('filtered ball vs GT'));
  await pv(() => window.__pv.setCandType('moved')); await p.waitForTimeout(200);
  check('S12 moved candidates match the export', await pv(() => document.querySelectorAll('#cands .ev').length === window.__pv.ds.meta.ballq.moved.length));
  check('S12 legend names the filtered ball', await pv(() => [...document.querySelectorAll('#legend span')].some(s => s.textContent.trim().startsWith('ball filtered'))));
} else check('S12 skipped (no ballq in export)', true);
await p.screenshot({ path: 'e2e.png' });
await b.close();
for (const [ok, name, info] of results) console.log((ok ? 'ok   ' : 'FAIL ') + name + (info ? '  (' + info + ')' : ''));
console.log(errs.length ? 'PAGE ERRORS:\n' + errs.join('\n') : 'no page errors');
process.exit(results.every(r => r[0]) && !errs.length ? 0 : 1);
