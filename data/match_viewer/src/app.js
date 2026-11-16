import { DataSet } from './data.js';
import { deriveEvents } from './events.js';
import { computePossession } from './possession.js';
import { AnnotationStore, SCHEMA, TOOL, TYPE } from './annotations.js';
import { CandidateQueue } from './queue.js';
import { rasterLabel, rasterMask, rasterIntensity, rasterCategorical, rasterInterval, rasterSlot, aggregateBands } from './timeline.js';
import { slotRuns, orderSlots } from './slots.js';
// app.js is the only file that knows about the page. It wires the modules together,
// draws on the two canvases and handles the keyboard. No analysis lives here.

const $ = s => document.querySelector(s);
const q = new URLSearchParams(location.search);
const C = getComputedStyle(document.documentElement);
const col = n => C.getPropertyValue('--' + n).trim();
const hex = h => { const v = parseInt(h.slice(1), 16); return [v >> 16 & 255, v >> 8 & 255, v & 255]; };
const TEAM = { H: col('home'), A: col('away'), D: col('dead'), X: col('none'), '': col('none'), '?': '#888' };
const TEAM3 = Object.fromEntries(Object.entries(TEAM).map(([k, v]) => [k, hex(v)]));
const BG3 = [34, 34, 34];
const rgb = a => `rgb(${a[2]},${a[1]},${a[0]})`;            // exporter colours are BGR
const fmt = v => Number.isFinite(v) ? v.toFixed(1) : 'n/a';
const lighten = (c, k) => c.map(v => Math.round(v + (255 - v) * k));
const STATE_SHADE = { controlled: 0, flight: 0.45, loose: 0.25, dead: null };

// ---------------------------------------------------------------- state
let ds, E, P, store, queue, exportDir;
let cur = 0, pinned = [], hover = null, layers = {}, view = [0, 0];
let armed = null, pendingClicks = [], magnify = false, loop = null;
let candFrames = [], candType = 'gt_vs_lh', includeHandled = false;
// clear() fires 'pv:changed' so an embedding page (match_dual.html) knows to redraw its strips
let ROWS = [], rowCache = new (class extends Map { clear() { super.clear(); window.dispatchEvent(new Event('pv:changed')); } })(), tlDrag = null, slotsOpen = false;
let RH = 11, OVH = 10, BANDH = 7;
const RG = 2, GG = 9, FLAG = 14;
// Things that can show up in the overview at the top of the timeline. GT always comes first.
// A strip just reuses one of the rows, a band is a per bin summary from aggregateBands.
const OVERVIEW_ITEMS = [
  { name: 'GT', kind: 'strip', row: 'GT' }, { name: 'PRED', kind: 'strip', row: 'PRED' }, { name: 'L&H', kind: 'strip', row: 'L&H' }, { name: 'L&H FILTERED', kind: 'strip', row: 'L&H FILTERED' }, { name: 'SEGMENT', kind: 'strip', row: 'SEGMENT' },
  { name: 'DEAD BALL', kind: 'strip', row: 'DEAD BALL' },
  { name: 'restarts', kind: 'band', key: 'restarts', count: true, color: 'good' }, { name: 'kicks', kind: 'band', key: 'kicks', count: true, color: 'kick' },
  { name: 'ball lost', kind: 'band', key: 'lostFrac', color: 'lost' }, { name: 'L&H share', kind: 'band', key: 'share' },
  { name: 'player count off', kind: 'band', key: 'countBad', color: 'warn' }, { name: 'homography bad', kind: 'band', key: 'homBad', color: 'bad' },
];
const OVERVIEW_DEFAULT = ['GT', 'DEAD BALL', 'restarts', 'ball lost', 'L&H share'];

const LAYERS = [
  ['pitch', 'pitch lines', true], ['players', 'players', true], ['holder', 'L&H holder disc', true], ['ring', 'possession ring', true],
  ['ball', 'ball (detector)', true], ['heat', 'ball probability map (darker = more likely)', true], ['ballq', 'ball (filtered) + 1 sigma ellipse', true], ['ballqlink', 'detector to filtered link', true], ['trail', 'ball trail (1.5 s)', true], ['future', 'ball future (1 s)', true], ['opp', 'closest opponent line', true],
  ['events', 'event marks (kick / jump / vanish)', true], ['referee', 'AR flag signals', true], ['ann', 'annotation marks', true], ['regions', 'dominant regions', false], ['hull', 'team hulls', false],
  ['arrows', 'attack direction', true], ['labels', 'slot numbers', true],
];
const LAYER_PRESETS = { 'everything': null, 'players + ball': ['pitch', 'players', 'ball', 'labels'], 'referee flags': ['pitch', 'referee'], 'clean video': [] };
const ROW_PRESETS = {
  'all': null,
    'possession': ['GT', 'PRED', 'NAIVE', 'L&H', 'L&H FILTERED', 'DISAGREE', 'DEAD OURS', 'SUBMISSION', 'SUB ≠ GT', 'SEGMENT', 'MINE', 'RESTART', 'KICK'],
  'submission': ['GT', 'DEAD OURS', 'L&H FILTERED', 'SUBMISSION', 'SUB ≠ GT', 'RESTART', 'BALL LOST'],
  'ball filter': ['GT', 'L&H', 'L&H FILTERED', 'DISAGREE', 'SUBMISSION', 'BALL LOST', 'BALL MOVED', 'BALL SD', 'TRUST', 'BALL AIR FLAG', 'WHITE BLEND'],
  'events': ['GT', 'RESTART', 'KICK', 'AR FLAGS', 'ANN EVENTS', 'BOOKMARK'],
  'data quality': ['GT', 'BALL LOST', 'BALL JUMP', 'PLAYER JUMP', 'DROPOUT <0.5 s', 'VANISH/APPEAR', 'COUNT ≠ 22', '› BOLT', 'HOMOGRAPHY', 'ANN DATA'],
  'annotation': ['GT', 'SEGMENT', 'MINE', 'ANN EVENTS', 'ANN DATA', 'SEGMENT B', 'ANN B', 'BOOKMARK'],
  'dead ball': ['GT', 'DEAD BALL', 'NO VIEW (HOM ∅ ∩ ≠22)', 'BALL LOST T∩B', 'NO COUNTERPART', 'RESTART', 'BALL LOST', 'COUNT ≠ 22', 'HOMOGRAPHY'],
  'minimal': ['GT', 'L&H', 'L&H FILTERED', 'RESTART', 'BALL LOST'],
};

// ---------------------------------------------------------------- boot
async function tryFetchJson(url) {
  try { const r = await fetch(url); if (!r.ok) return null; return await r.json(); } catch (e) { return null; }
}
/** Map raw referee_signals.json detections → viewer event records. */
function refereeDocToEvents(doc) {
  const styles = (ds && ds.meta.referee && ds.meta.referee.styles) || {
    none: { glyph: 'cross', color: [80, 80, 80], hex: '#ffdc00' },
    unknown: { glyph: 'cross', color: [120, 120, 120], hex: '#ffdc00' },
    attention_raised: { glyph: 'tri_up', hex: '#ffdc00' },
    throw_in_left: { glyph: 'diamond', hex: '#00a0ff' },
    throw_in_right: { glyph: 'diamond', hex: '#ffa000' },
    corner: { glyph: 'circle', hex: '#ff5050' },
    goal_kick: { glyph: 'square', hex: '#50c850' },
    substitution: { glyph: 'star', hex: '#dc64dc' },
  };
  const out = [];
  for (const d of (doc.detections || [])) {
    const sig = d.signal || 'none';
    const st = styles[sig] || styles.unknown || { hex: '#ffdc00', glyph: 'cross', color: [0, 220, 255] };
    out.push({
      f: d.f, abs_f: d.f, official_id: d.official_id, signal: sig, confidence: d.confidence || 0,
      bbox_xyxy: d.bbox_xyxy, flag_centroid: d.flag_centroid, flag_angle_deg: d.flag_angle_deg, rel: d.rel,
      glyph: st.glyph || 'cross', color: st.color || [0, 220, 255], hex: st.hex || '#ffdc00',
    });
  }
  return out;
}
async function loadRefereeSidecars() {
  // Sidecar JSON is authoritative for boxes (includes signal=none). Falls back to meta.referee.
  const src = (ds.meta.source || 'tactical');
  const candidates = [
    exportDir + '/referee_signals.json',
    'export/referee_signals_' + src + '.json',
    'export/referee_signals_tactical.json',
    'export/referee_signals.json',
    'referee_signals_' + src + '.json',
  ];
  for (const url of candidates) {
    const doc = await tryFetchJson(url);
    if (!doc || !doc.detections || !doc.detections.length) continue;
    E.refereeByFrame = new Map();
    E.refereeFrames = [];
    E.refereeSignalFrames = [];
    E.ingestRefereeEvents(refereeDocToEvents(doc), { absolute: true });
    if (!ds.meta.referee) ds.meta.referee = { source: doc.source || src, styles: null, events: [] };
    console.info('[referee] loaded', url, '→', E.refereeFrames.length, 'frames with boxes');
    return;
  }
  if (!E.refereeFrames.length) console.warn('[referee] no meta.referee and no sidecar JSON found');
}
const DEAD_BALL_HINT = 'python dead_ball_investigation/run.py --from-exports';
async function loadDeadBall() {
  // written by dead_ball_investigation/run.py; holds intervals for both timelines
  const src = ds.meta.source || 'tactical';
  E.deadMissing = 'no export/dead_ball.json';
  for (const url of [exportDir + '/dead_ball.json', 'export/dead_ball.json']) {
    const doc = await tryFetchJson(url);
    if (!doc) continue;
    if (E.ingestDeadBall(doc, src)) { E.deadMissing = null; ds.deadMask = E.dead.mask; ds.resetSubmission(); console.info('[dead ball] loaded', url, '→', E.deadStarts.length, 'intervals in this export'); return; }   // the mask also feeds the SUBMISSION row
    E.deadMissing = `${url} has no ${src} timeline`;
  }
  console.warn('[dead ball]', E.deadMissing, '— run:', DEAD_BALL_HINT);
}
async function load() {
  exportDir = q.get('d');
  if (!exportDir) { try { exportDir = (await (await fetch('export/latest.json')).json()).dir; } catch (e) { exportDir = 'export/tactical_gt'; } }
  $('#srcName').textContent = 'loading ' + exportDir + ' …';
  const meta = await (await fetch(exportDir + '/meta.json')).json();
  const buf = await (await fetch(exportDir + '/data.bin')).arrayBuffer();
  let bq = null;
  if (meta.ballq) { try { bq = new Float32Array(await (await fetch(exportDir + '/ballq.bin')).arrayBuffer()); } catch (e) { bq = null; } }
  ds = new DataSet(meta, new Float32Array(buf), bq);
  if (meta.ballq && meta.ballq.grid) {
    try { const gi = new Uint32Array(await (await fetch(exportDir + '/grid_index.bin')).arrayBuffer()); ds.setGridIndex(exportDir + '/grid.bin', gi, r => { if (r === cur) draw(); }); } catch (e) { /* no grid files: the layer stays empty */ }
  }
  E = deriveEvents(ds);
  await loadRefereeSidecars();
  await loadDeadBall();
  store = new AnnotationStore({ get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} } }, 'pv2:' + exportDir, ds);
  const nRef = E.refereeFrames.length;
  $('#srcName').textContent = `${meta.source} · frames ${meta.start}-${meta.end} · ${(ds.n / ds.fps / 60).toFixed(1)} min` + (nRef ? ` · AR boxes on ${nRef} frames` : '') + (E.dead ? ` · ${E.deadStarts.length} dead-ball intervals` : ` · no dead-ball rows (${E.deadMissing}; run ${DEAD_BALL_HINT})`);
  document.title = `Match viewer · ${meta.source}`;
  const v = $('#v'); v.src = meta.video;
  v.addEventListener('loadedmetadata', init, { once: true });
}

function init() {
  const v = $('#v'), ov = $('#ov');
  ov.width = v.videoWidth; ov.height = v.videoHeight;
  P = computePossession(ds, E.kick, +$('#rad').value);
  view = [0, ds.n - 1];
  buildLayers(); buildPalette(); buildLegend(); buildRows(); setupTimeline(); setupRowFilter(); setupOverviewFilter(); setupHelp();
  buildCandidates(); renderBookmarks(); renderAnn(); renderAgreement();
  fit(); new ResizeObserver(fit).observe($('#stage')); window.addEventListener('resize', fit); setTimeout(fit, 300);
  if ('requestVideoFrameCallback' in v) { const cb = (now, md) => { onTime(md.mediaTime); v.requestVideoFrameCallback(cb); }; v.requestVideoFrameCallback(cb); }
  else v.addEventListener('timeupdate', () => onTime(v.currentTime));
  v.addEventListener('seeked', () => onTime(v.currentTime));
  v.addEventListener('play', () => $('#bPlay').textContent = '❚❚ pause'); v.addEventListener('pause', () => $('#bPlay').textContent = '▶ play');
  $('#bPlay').onclick = togglePlay; $('#speed').onchange = e => v.playbackRate = +e.target.value;
  $('#rad').oninput = e => { $('#radV').textContent = (+e.target.value).toFixed(1) + ' m'; P = computePossession(ds, E.kick, +e.target.value); rowCache.clear(); buildCandidates(); renderAgreement(); draw(); drawTimeline(); };
  $('#bPrev').onclick = () => stepEvent(-1); $('#bNext').onclick = () => stepEvent(1);
  $('#candType').onchange = e => { candType = e.target.value; buildCandidates(); };
  $('#inclHandled').onchange = e => { includeHandled = e.target.checked; };
  $('#bAdd').onclick = addBookmark; $('#bExpB').onclick = () => download(`bookmarks_${ds.meta.source}.csv`, store.exportBookmarks());
  $('#bExpHR').onclick = () => download(`human_reading_${ds.meta.source}_${ds.start}-${ds.end}.csv`, store.exportHumanReading(ds.meta.source));
  $('#bExpA').onclick = () => download(`annotations_${ds.meta.source}_${ds.start}-${ds.end}.json`, store.exportJSON(ds.meta.source));
  $('#bExpAC').onclick = () => download(`annotations_${ds.meta.source}_${ds.start}-${ds.end}.csv`, store.exportCSV());
  $('#impA').onchange = importAnn; $('#bClearB').onclick = () => { store.clearLayerB(); rowCache.clear(); renderAgreement(); drawTimeline(); draw(); };
  $('#bClr').onclick = () => { if (confirm('clear my annotations, segments and bookmarks for this export?')) { store.segments = []; store.points = []; store.bookmarks = []; store._save(); rowCache.clear(); renderAll(); } };
  $('#annFilter').onchange = renderAnn;
  $('#note').addEventListener('input', noteIndicator);
  ov.addEventListener('mousemove', onPitchHover); ov.addEventListener('mouseleave', () => { hover = null; $('#tip').style.display = 'none'; $('#mag').style.display = 'none'; draw(); });
  ov.addEventListener('click', onPitchClick);
  document.addEventListener('keydown', onKey);
  seek(0);
  window.__pv.ready = true; window.dispatchEvent(new Event('pv:ready'));
}
function renderAll() { buildRows(); buildCandidates(); renderBookmarks(); renderAnn(); renderAgreement(); drawTimeline(); draw(); updateInfo(); }

function fit() {
  const st = $('#stage'), v = $('#v'), ov = $('#ov'), wrap = $('#wrap'); if (!v.videoWidth) return;
  const ar = v.videoWidth / v.videoHeight, sw = st.clientWidth - 8, sh = st.clientHeight - 8;
  let w = sw, h = w / ar; if (h > sh) { h = sh; w = h * ar; } w = Math.floor(w); h = Math.floor(h);
  wrap.style.width = v.style.width = ov.style.width = w + 'px'; wrap.style.height = v.style.height = ov.style.height = h + 'px';
  applyZoom(); placeCards();
  draw();
}

// ---------------------------------------------------------------- layout: video zoom, draggable cards, splitters
// Persisted per origin (each --port instance keeps its own layout).
const EMBED = !!q.get('embed');
const LAYOUT_KEY = EMBED ? 'pv:layout:embed:' + (q.get('d') || '') : 'pv:layout';
const layout = (() => { try { return JSON.parse(localStorage.getItem(LAYOUT_KEY)) || {}; } catch (e) { return {}; } })();
const saveLayout = () => { try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)); } catch (e) {} };
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
let vzS = 1, vzX = 0, vzY = 0, pan = null, suppressClick = false;

function applyZoom() {
  const wrap = $('#wrap'), W = wrap.clientWidth, H = wrap.clientHeight;
  vzS = clamp(vzS, 1, 10); vzX = clamp(vzX, W - W * vzS, 0); vzY = clamp(vzY, H - H * vzS, 0);
  $('#vz').style.transform = vzS === 1 ? '' : `translate(${vzX}px,${vzY}px) scale(${vzS})`;
  $('#zoomV').textContent = vzS.toFixed(1) + '×';
}
// px, py: point in #wrap coordinates that stays fixed while zooming
function zoomAt(px, py, k) {
  const s = clamp(vzS * k, 1, 10); if (s === vzS) return;
  vzX = px - (px - vzX) * s / vzS; vzY = py - (py - vzY) * s / vzS; vzS = s; applyZoom();
}
function zoomCenter(k) { const wrap = $('#wrap'); zoomAt(wrap.clientWidth / 2, wrap.clientHeight / 2, k); }
function resetZoom() { vzS = 1; vzX = vzY = 0; applyZoom(); }
const toolNeedsClick = () => armed || (store && store.pending && store.pending.x !== undefined);

function setupVideoZoom() {
  const clip = $('#vclip');
  clip.addEventListener('wheel', e => {
    e.preventDefault(); const r = $('#wrap').getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-clamp(e.deltaY, -200, 200) * 0.002));
  }, { passive: false });
  clip.addEventListener('mousedown', e => { if (e.button === 1) e.preventDefault(); });   // no autoscroll on middle click
  clip.addEventListener('contextmenu', e => { if (vzS > 1 || pan) e.preventDefault(); });
  clip.addEventListener('pointerdown', e => {
    const left = e.button === 0 && vzS > 1 && !toolNeedsClick();
    if (!(left || e.button === 1 || e.button === 2) || vzS === 1) return;
    pan = { x0: e.clientX, y0: e.clientY, vx: vzX, vy: vzY, moved: false, btn: e.button };
  });
  window.addEventListener('pointermove', e => {
    if (!pan) return;
    const dx = e.clientX - pan.x0, dy = e.clientY - pan.y0;
    if (!pan.moved && Math.hypot(dx, dy) < 4) return;
    pan.moved = true; clip.classList.add('panning');
    vzX = pan.vx + dx; vzY = pan.vy + dy; applyZoom();
  });
  window.addEventListener('pointerup', () => {
    if (!pan) return;
    if (pan.moved && pan.btn === 0) { suppressClick = true; setTimeout(() => suppressClick = false, 0); }
    pan = null; clip.classList.remove('panning');
  });
  $('#bZoomIn').onclick = () => zoomCenter(1.25); $('#bZoomOut').onclick = () => zoomCenter(0.8); $('#bZoomReset').onclick = resetZoom;
}

// card position is stored as a fraction of the video box so it survives resizes
function placeCards() {
  const el = $('#possCard'), wrap = $('#wrap'), c = layout.card || {};
  el.classList.toggle('vert', !!c.vert);
  if (c.fx === undefined) { el.style.left = '12px'; el.style.top = '12px'; return; }
  const W = wrap.clientWidth, H = wrap.clientHeight;
  el.style.left = clamp(c.fx * W, 0, Math.max(0, W - el.offsetWidth)) + 'px';
  el.style.top = clamp(c.fy * H, 0, Math.max(0, H - el.offsetHeight)) + 'px';
}
function setupCardDrag() {
  const el = $('#possCard'), grip = $('#possGrip');
  grip.addEventListener('pointerdown', e => {
    if (e.button !== 0 || e.target.closest('button')) return;
    e.preventDefault(); grip.setPointerCapture(e.pointerId); el.classList.add('dragging');
    const wr = $('#wrap').getBoundingClientRect(), ox = e.clientX - el.offsetLeft, oy = e.clientY - el.offsetTop;
    const move = ev => {
      const x = clamp(ev.clientX - ox, 0, Math.max(0, wr.width - el.offsetWidth)), y = clamp(ev.clientY - oy, 0, Math.max(0, wr.height - el.offsetHeight));
      el.style.left = x + 'px'; el.style.top = y + 'px';
      layout.card = { ...(layout.card || {}), fx: x / wr.width, fy: y / wr.height };
    };
    const up = () => { grip.removeEventListener('pointermove', move); grip.removeEventListener('pointerup', up); el.classList.remove('dragging'); saveLayout(); };
    grip.addEventListener('pointermove', move); grip.addEventListener('pointerup', up);
  });
  grip.addEventListener('dblclick', e => { if (e.target.closest('button')) return; const v = layout.card && layout.card.vert; layout.card = v ? { vert: v } : {}; saveLayout(); placeCards(); });
  $('#bCardDir').onclick = () => { layout.card = { ...(layout.card || {}), vert: !(layout.card && layout.card.vert) }; saveLayout(); placeCards(); };
}

// generic drag-to-resize on a splitter element
function splitter(el, onMove, onReset) {
  el.addEventListener('pointerdown', e => {
    if (e.button !== 0) return; e.preventDefault(); el.setPointerCapture(e.pointerId);
    el.classList.add('on'); document.body.classList.add('resizing');
    const move = ev => onMove(ev);
    const up = () => { el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up); el.classList.remove('on'); document.body.classList.remove('resizing'); saveLayout(); };
    el.addEventListener('pointermove', move); el.addEventListener('pointerup', up);
  });
  el.addEventListener('dblclick', () => { onReset(); saveLayout(); });
}
function setSideW(w) { if (w) document.body.style.setProperty('--sidew', w + 'px'); else document.body.style.removeProperty('--sidew'); }
function setTlH(h) { const tl = $('#tl'); tl.classList.toggle('sized', !!h); tl.style.height = h ? h + 'px' : ''; }
function setLblW(w) { if (w) $('#tl').style.setProperty('--lblw', w + 'px'); else $('#tl').style.removeProperty('--lblw'); }
function setRowH(h) { RH = h; OVH = Math.max(10, h - 1); BANDH = Math.max(7, Math.round(h * 0.75)); $('#rowH').value = h; if (ROWS.length) drawTimeline(); }
function setPanels() {
  const side = layout.side ?? !EMBED, tl = layout.tl ?? !EMBED;
  document.body.classList.toggle('noside', !side); document.body.classList.toggle('notl', !tl);
  $('#bTogSide').classList.toggle('off', !side); $('#bTogTl').classList.toggle('off', !tl);
}
function setupLayout() {
  document.body.classList.toggle('embed', EMBED);
  setSideW(layout.sideW); setTlH(layout.tlH); setLblW(layout.lblW); setRowH(layout.rowH || 11); setPanels();
  $('#bTogSide').onclick = () => { layout.side = document.body.classList.contains('noside'); saveLayout(); setPanels(); };
  $('#bTogTl').onclick = () => { layout.tl = document.body.classList.contains('notl'); saveLayout(); setPanels(); };
  splitter($('#sideSplit'), e => { layout.sideW = Math.round(clamp(window.innerWidth - e.clientX - 2, 200, window.innerWidth * 0.7)); setSideW(layout.sideW); }, () => { delete layout.sideW; setSideW(); });
  splitter($('#tlSplit'), e => { layout.tlH = Math.round(clamp(window.innerHeight - e.clientY, 80, window.innerHeight - 160)); setTlH(layout.tlH); }, () => { delete layout.tlH; setTlH(); });
  splitter($('#lblSplit'), e => { layout.lblW = Math.round(clamp(e.clientX - $('#rowlabels').getBoundingClientRect().left, 40, 420)); setLblW(layout.lblW); }, () => { delete layout.lblW; setLblW(); });
  $('#rowH').oninput = e => { layout.rowH = +e.target.value; setRowH(layout.rowH); saveLayout(); };
  setupVideoZoom(); setupCardDrag();
}

// ---------------------------------------------------------------- navigation
const frameToTime = r => (ds.start + r + 0.5) / ds.fps;
function onTime(t) {
  const r = Math.max(0, Math.min(ds.n - 1, Math.floor(t * ds.fps) - ds.start));
  if (loop && r > loop[1]) { $('#v').currentTime = frameToTime(loop[0]); return; }
  cur = r; draw(); drawTimeline(); updateInfo();
}
function seek(r) { r = Math.max(0, Math.min(ds.n - 1, r)); $('#v').currentTime = frameToTime(r); cur = r; draw(); drawTimeline(); updateInfo(); }
function togglePlay() { const v = $('#v'); v.paused ? v.play() : v.pause(); }
function stepEvent(dir) {
  const type = $('#stepType').value;
  if (type === 'cand') { const t = queue.next(cur, dir, { includeHandled }); if (t !== null) seek(t); return; }
  const L = E.list(type); if (!L.length) return;
  let t = dir > 0 ? L.find(r => r > cur) : [...L].reverse().find(r => r < cur);
  if (t === undefined) t = dir > 0 ? L[0] : L[L.length - 1]; seek(t);
}
function toggleLoop() { if (loop) { loop = null; } else { loop = [Math.max(0, cur - 2 * ds.fps), cur]; seek(loop[0]); $('#v').play(); } $('#loopInd').textContent = loop ? `loop ${ds.start + loop[0]}-${ds.start + loop[1]}` : ''; }

function onKey(e) {
  const tag = e.target.tagName; const inNote = e.target.id === 'note';
  if (inNote && e.key === 'Escape') { e.target.value = ''; noteIndicator(); e.target.blur(); return; }
  if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT') return;
  const v = $('#v'), step = e.shiftKey ? Math.round(ds.fps) : 1, K = e.key;
  if ((e.ctrlKey || e.metaKey) && K.toLowerCase() === 'z') { e.preventDefault(); if (store.undo()) { rowCache.clear(); renderAll(); } return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  // if a tool is armed or an interval is open, those keys win
  if (store.pending || armed) {
    if (K === 'Escape') { cancelTool(); return; }
    if (K === 'Enter') { if (store.pending && readyToClose()) { closePending(); } return; }
    if (store.pending && store.pending.x !== undefined && K.startsWith('Arrow')) { e.preventDefault(); const d = e.shiftKey ? 0.5 : 0.1; store.nudgePending(K === 'ArrowRight' ? d : K === 'ArrowLeft' ? -d : 0, K === 'ArrowDown' ? d : K === 'ArrowUp' ? -d : 0); draw(); return; }
    if (armed && armed.need === 'pitch+team' && pendingClicks.length === 1 && 'hHaA'.includes(K)) { finishTool({ team: K.toUpperCase() }); return; }
    if (store.pending && TOOL[K.toUpperCase()] && TOOL[K.toUpperCase()].type === store.pending.type) { closePending(); return; }
  }
  if (K === ' ') { e.preventDefault(); togglePlay(); }
  else if (K === 'ArrowRight') { e.preventDefault(); v.pause(); seek(cur + step); }
  else if (K === 'ArrowLeft') { e.preventDefault(); v.pause(); seek(cur - step); }
  else if (K === ',') stepEvent(-1); else if (K === '.') stepEvent(1);
  else if (K === '[' || K === ']') { const s = $('#speed'); s.selectedIndex = Math.max(0, Math.min(s.options.length - 1, s.selectedIndex + (K === ']' ? 1 : -1))); v.playbackRate = +s.value; }
  else if (K === 'b' || K === 'B') addBookmark();
  else if (K === 'Enter') { markCandidate('done'); } else if (K === 'Backspace') { e.preventDefault(); markCandidate('skipped'); } else if (K === '!') markCandidate('flagged');
  else if (K === 'h' || K === 'H') setSeg({ team: 'H' }); else if (K === 'a' || K === 'A') setSeg({ team: 'A' });
  else if (K === 'd' || K === 'D' || K === '4') setSeg({ state: 'dead' }); else if (K === 'x' || K === 'X' || K === '0') setSeg(null);
  else if (K === '1') setSeg({ state: 'controlled' }); else if (K === '2') setSeg({ state: 'flight' }); else if (K === '3') setSeg({ state: 'loose' });
  else if (K === 'z' || K === 'Z') { magnify = !magnify; $('#mag').style.display = magnify ? 'block' : 'none'; }
  else if (K === 'l' || K === 'L') toggleLoop();
  else if (K === '=' || K === '+') zoomCenter(1.25); else if (K === '-') zoomCenter(0.8);
  else if (K === '?') toggleHelp();
  else if (K === 'Escape') { if ($('#help').style.display === 'block') toggleHelp(); else { view = [0, ds.n - 1]; pinned = []; loop = null; resetZoom(); $('#loopInd').textContent = ''; renderPinned(); drawTimeline(); draw(); } }
  else if (K === 'Home') seek(0); else if (K === 'End') seek(ds.n - 1);
  else if (TOOL[K.toUpperCase()] && TOOL[K.toUpperCase()].cat !== 'state') arm(K.toUpperCase());
}

// ---------------------------------------------------------------- segments (S4)
function setSeg(change) { store.setSegment(cur, change); rowCache.clear(); drawTimeline(); draw(); updateInfo(); }

// ---------------------------------------------------------------- tools, intervals (S2, S5)
function arm(key) {
  const t = TOOL[key]; if (!t) return;
  $('#v').pause();
  if (t.cat === 'event') { takeNote(store.addPoint(cur, { cat: 'event', type: t.type, ...ballPos() })); rowCache.clear(); renderAnn(); drawTimeline(); draw(); return; }
  if (!t.need) { store.beginInterval(cur, { cat: t.cat, type: t.type }); armed = null; showPending(); return; }
  if (t.type === 'ball_pos') { const b = ds.ball(cur); if (b) { store.beginInterval(cur, { cat: t.cat, type: t.type, x: b.x, y: b.y, bx: b.x, by: b.y }); armed = null; $('#armed').textContent = 'true ball position: placed at the tracked ball. Arrows nudge 10 cm (Shift 50 cm), click to move, N or Enter to close'; draw(); return; } }
  armed = t; pendingClicks = [];
  $('#armed').textContent = t.need === 'pitch' ? `${t.label}: click the spot on the pitch` : t.need === 'pitch+team' ? `${t.label}: click the spot, then press H or A` : t.need === 'player2' ? `${t.label}: click the two markers` : `${t.label}: click the marker`;
  paletteState();
}
function ballPos() { const b = ds.ball(cur); return b ? { bx: b.x, by: b.y } : {}; }
function takeNote(a) { const n = $('#note').value.trim(); if (n) { a.note = n; $('#note').value = ''; noteIndicator(); store._save(); } return a; }
function readyToClose() { const p = store.pending; return !!p && !(armed); }
function showPending() { const p = store.pending; $('#armed').textContent = p ? `${TYPE[p.type].label}: open since frame ${ds.start + p.f}. Press ${keyOf(p.type)} or Enter to close, Esc to discard` : ''; paletteState(); drawTimeline(); draw(); }
function keyOf(type) { return Object.keys(TOOL).find(k => TOOL[k].type === type); }
function closePending() { const p = store.endInterval(cur); if (p) takeNote(p); $('#armed').textContent = ''; rowCache.clear(); paletteState(); renderAnn(); drawTimeline(); draw(); }
function cancelTool() { armed = null; pendingClicks = []; store.discardPending(); $('#armed').textContent = ''; paletteState(); drawTimeline(); draw(); }
function paletteState() { document.querySelectorAll('.tool').forEach(el => el.classList.toggle('armed', (armed && el.dataset.key === keyOf(armed.type)) || (store.pending && el.dataset.key === keyOf(store.pending.type)))); }
function toolClick(e) {
  const [px, py] = pitchXY(e);
  if (store.pending && store.pending.x !== undefined && !armed) { const p = ds.pitchFromPx(cur, px, py); if (p) { store.pending.x = p.x; store.pending.y = p.y; draw(); } return; }
  if (!armed) return;
  if (armed.need === 'pitch' || armed.need === 'pitch+team') {
    const p = ds.pitchFromPx(cur, px, py); if (!p) { $('#armed').textContent = 'no homography on this frame, cannot place'; return; }
    pendingClicks = [p]; if (armed.need === 'pitch') finishTool({ x: p.x, y: p.y }); else { $('#armed').textContent = `${armed.label}: now press H (${ds.teamName('H')}) or A (${ds.teamName('A')})`; draw(); } return;
  }
  const h = nearestHit(e); if (!h) return;
  pendingClicks.push(h.k);
  if (armed.need === 'player' || pendingClicks.length === 2) finishTool({ slots: pendingClicks.slice() }); else $('#armed').textContent = `${armed.label}: click the second marker`;
}
function finishTool(extra) {
  const a = { cat: armed.cat, type: armed.type, ...extra, ...ballPos() };
  if (pendingClicks.length && pendingClicks[0].x !== undefined) { a.x = pendingClicks[0].x; a.y = pendingClicks[0].y; }
  store.beginInterval(cur, a); armed = null; pendingClicks = []; showPending();
}

// ---------------------------------------------------------------- candidates and queue (S1)
function segmentStarts(pred) { const out = []; let prev = false; for (let r = 0; r < ds.n; r++) { const c = pred(r); if (c && !prev) out.push(r); prev = c; } return out; }
function candidateList(t) {
  const gt = r => ds.gt(r), alive = r => gt(r) === 'H' || gt(r) === 'A', tp = +$('#rad').value;
  const ringCount = r => { const b = ds.ball(r); if (!b) return -1; let n = 0; for (const p of ds.players(r)) if (ds.metres(b, p) <= tp) n++; return n; };
  const L = {
    gt_vs_lh: () => ds.meta.gt ? segmentStarts(r => alive(r) && P.lh[r] && P.lh[r] !== gt(r)) : [],
    gt_vs_pred: () => (ds.meta.gt && ds.meta.pred) ? segmentStarts(r => gt(r) !== 'X' && ds.pred(r) !== gt(r)) : [],
    gt_change: () => E.gtChanges, restart: () => E.restartFrames, kick: () => E.list('kick'), referee: () => E.list('referee'),
    lost: () => E.lostStarts, jump: () => E.jumpFrames, churn: () => E.churnFrames, dead: () => E.list('dead'),
    ambig: () => segmentStarts(r => alive(r) && ringCount(r) !== -1 && ringCount(r) !== 1),
    lh_raw_vs_post: () => ds.hasBallq() ? segmentStarts(r => P.lh[r] && ds.lhPost(r) && ds.lhPost(r) !== 'X' && P.lh[r] !== ds.lhPost(r)) : [],
    moved: () => ds.hasBallq() ? ds.meta.ballq.moved.map(m => m[0]) : [],
    gt_vs_sub: () => ds.hasDeadOurs() ? segmentStarts(r => { const g = ds.gt(r); return (g === 'H' || g === 'A' || g === 'D') && ds.submission(r) !== g; }) : [],
    dead_vs_gt: () => ds.hasDeadOurs() ? segmentStarts(r => { const g = ds.gt(r); return (g === 'D') !== ds.deadOurs(r) && (g === 'H' || g === 'A' || g === 'D'); }) : [],
  };
  const label = {
    gt_vs_lh: r => `GT ${gt(r)} · L&H ${P.lh[r]}`, gt_vs_pred: r => `GT ${gt(r)} · pred ${ds.pred(r)}`, gt_change: r => `${gt(r - 1)} → ${gt(r)}`,
    restart: r => { const f = E.flags.find(x => x.r === r); return f ? f.name : 'GT dead → alive'; }, kick: r => `ball ${ds.ball(r) ? ds.ball(r).sf.toFixed(0) : '?'} m/s`,
    referee: r => { const dets = E.refereeByFrame.get(r) || []; return dets.map(d => `${d.official_id || 'AR'} ${d.signal}`).join(' · ') || 'flag'; },
    lost: r => { let e = r; while (e < ds.n && !ds.ball(e)) e++; return `${((e - r) / ds.fps).toFixed(1)} s`; },
    dead: r => { const D = E.dead; let e = r, v = 0, bl = 0; while (e < ds.n && D.mask[e]) { v += D.noView[e]; bl += D.ballLost[e]; e++; } return `${((e - r) / ds.fps).toFixed(1)} s · ${v && bl ? 'both terms' : v ? 'no view (B)' : 'ball lost T∩B'}`; },
    jump: r => { const b = ds.meta.events.bjump.find(e => e[0] === r); if (b) return `ball ${b[1].toFixed(1)} m`; const p = ds.meta.events.pjump.find(e => e[0] === r); return p ? `#${ds.numOf(p[1])} ${p[2].toFixed(1)} m` : ''; },
    churn: r => [...ds.meta.events.vanish.filter(e => e[0] === r).map(e => 'lost #' + ds.numOf(e[1])), ...ds.meta.events.appear.filter(e => e[0] === r).map(e => 'new #' + ds.numOf(e[1]))].join(', '),
    ambig: r => `${ringCount(r)} within ${tp} m`,
    lh_raw_vs_post: r => `raw ${P.lh[r]} · filtered ${ds.lhPost(r)} · GT ${gt(r) || '-'}`,
    moved: r => { const m = ds.meta.ballq.moved.find(x => x[0] === r); return m ? `${m[1].toFixed(1)} m` : ''; },
    gt_vs_sub: r => `GT ${gt(r)} · submission ${ds.submission(r) || '-'}`,
    dead_vs_gt: r => ds.deadOurs(r) ? `we say dead, GT ${gt(r)}` : `GT dead, we say ${ds.submission(r) || '-'}`,
  };
  return { frames: (L[t] || (() => []))(), label: label[t] || (() => '') };
}
function buildCandidates() {
  const { frames, label } = candidateList(candType); candFrames = frames;
  queue = new CandidateQueue(frames, store.savedQueue(candType));
  const el = $('#cands'); el.innerHTML = '';
  for (const r of frames.slice(0, 600)) {
    const d = document.createElement('div'); d.className = 'ev'; d.dataset.r = r;
    d.innerHTML = `<span class="st"></span><span class="f mono">${ds.start + r}</span><span class="t">${label(r)}</span>`;
    d.onclick = () => seek(r); el.appendChild(d);
  }
  renderQueue(); drawTimeline();
}
function renderQueue() {
  const p = queue.progress();
  $('#candN').textContent = `done ${p.done} · skipped ${p.skipped} · flagged ${p.flagged} · ${p.total} total`;
  for (const d of $('#cands').children) { const s = queue.statusOf(+d.dataset.r); d.querySelector('.st').className = 'st ' + s; d.querySelector('.st').textContent = s === 'done' ? '✓' : s === 'skipped' ? '›' : s === 'flagged' ? '!' : ''; d.classList.toggle('cur', +d.dataset.r === cur); }
}
function markCandidate(status) {
  if (!candFrames.length) return;
  const f = candFrames.includes(cur) ? cur : queue.next(cur, -1, { includeHandled: true });
  if (f === null) return;
  const nxt = queue.markAndAdvance(f, status); store.saveQueue(candType, queue.serialize()); renderQueue();
  if (nxt !== null) seek(nxt);
}

// ---------------------------------------------------------------- bookmarks, notes
function addBookmark() { store.addBookmark(cur, $('#note').value.trim(), { gt: ds.gt(cur) || '', lh: P.lh[cur] }); $('#note').value = ''; noteIndicator(); rowCache.clear(); renderBookmarks(); drawTimeline(); }
function renderBookmarks() {
  const el = $('#bms'); el.innerHTML = '';
  store.bookmarks.forEach((b, i) => {
    const d = document.createElement('div'); d.className = 'ev';
    d.innerHTML = `<span class="pill" style="background:${TEAM[b.gt] || '#888'}">${b.gt || '-'}</span><span class="t">${b.note || '<span class="hint">no note</span>'}</span><span class="f mono">${ds.start + b.f}</span><span class="f" title="remove">✕</span>`;
    d.querySelector('.t').onclick = () => seek(b.f); d.lastElementChild.onclick = () => { store.removeBookmark(i); rowCache.clear(); renderBookmarks(); drawTimeline(); };
    el.appendChild(d);
  });
}
function noteIndicator() { const n = $('#note').value.trim(); $('#noteInd').textContent = n ? 'attached to the next annotation or bookmark (Esc clears)' : ''; }
function download(name, text) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' })); a.download = name; a.click(); }

// ---------------------------------------------------------------- annotations list, import, agreement (S3)
const annColor = a => (TYPE[a.type] || {}).color || '#fff', annLabel = a => (TYPE[a.type] || {}).label || a.type;
function renderAnn() {
  const f = $('#annFilter').value; const L = store.items('A').filter(a => !f || a.cat === f);
  $('#annN').textContent = `(${L.length})`; const el = $('#annList'); el.innerHTML = '';
  for (const a of L) {
    const d = document.createElement('div'); d.className = 'ev' + (a.f <= cur && cur <= (a.f2 ?? a.f) ? ' cur' : '');
    const span = a.f2 !== undefined && a.f2 !== a.f ? ` <span class="hint">${store.durationS(a).toFixed(1)} s</span>` : '';
    const extra = [a.team ? (a.team === 'H' ? 'BAR' : 'ATH') : '', a.slots ? a.slots.map(k => '#' + ds.numOf(k)).join(' ↔ ') : '', a.x !== undefined ? `(${a.x.toFixed(3)}, ${a.y.toFixed(3)})` : ''].filter(Boolean).join(' ');
    d.innerHTML = `<span class="g" style="background:${annColor(a)};border-radius:2px"></span><span class="t">${annLabel(a)}${span}${extra ? ' <span class="hint">' + extra + '</span>' : ''}${a.note ? '<div class="hint">' + a.note + '</div>' : ''}</span><span class="f mono">${ds.start + a.f}</span><span class="f" title="remove">✕</span>`;
    d.querySelector('.t').onclick = () => seek(a.f); d.lastElementChild.onclick = () => { store.remove(a.id); rowCache.clear(); renderAnn(); drawTimeline(); draw(); };
    el.appendChild(d);
  }
}
function importAnn(e) {
  const fl = e.target.files[0]; if (!fl) return; const rd = new FileReader();
  rd.onload = () => { try { const j = JSON.parse(rd.result); const asB = confirm('OK = load as layer B (second annotator, read-only)\nCancel = replace my annotations'); store.importLayer(j, asB ? 'B' : 'A'); rowCache.clear(); renderAll(); } catch (err) { alert('could not read: ' + err); } };
  rd.readAsText(fl); e.target.value = '';
}
// GT agreement of the raw and the filtered Link and Hoernig labels over the labelled frames
export function lhAgreement(ds, lhRaw) {
  let n = 0, raw = 0, post = 0, both = 0, onlyPost = 0, onlyRaw = 0, nd = 0, sub = 0, deadHit = 0, deadGt = 0, deadPred = 0;
  for (let r = 0; r < ds.n; r++) {
    const g = ds.gt(r);
    if (g === 'H' || g === 'A' || g === 'D') { nd++; if (ds.submission(r) === g) sub++; if (g === 'D') deadGt++; if (ds.deadOurs(r)) { deadPred++; if (g === 'D') deadHit++; } }
    if (g !== 'H' && g !== 'A') continue; n++; const a = lhRaw[r] === g, b = ds.lhPost(r) === g; raw += a; post += b; both += a && b; onlyPost += b && !a; onlyRaw += a && !b;
  }
  return { n, raw: n ? raw / n : NaN, post: n ? post / n : NaN, onlyPost, onlyRaw, nd, sub: nd ? sub / nd : NaN, deadRecall: deadGt ? deadHit / deadGt : NaN, deadPrecision: deadPred ? deadHit / deadPred : NaN };
}
let AG = null;   // GT agreement of both heuristics over the export, shown in the heuristics card and the agreement panel
function renderAgreement() {
  const el = $('#agree'); let html = '';
  AG = (ds.hasBallq() && ds.meta.gt) ? lhAgreement(ds, P.lh) : null;
  if (AG && AG.n) {
    const ag = AG, q = ds.meta.ballq.quality || {};
    html += `<div class="row"><span class="k">L&amp;H raw ball vs GT</span><span class="v">${(100 * ag.raw).toFixed(1)} %</span></div><div class="row"><span class="k">L&amp;H filtered ball vs GT</span><span class="v" style="color:${ag.post >= ag.raw ? col('good') : col('bad')}">${(100 * ag.post).toFixed(1)} %</span></div><div class="row"><span class="k">fixed / broken frames</span><span class="v">${ag.onlyPost} / ${ag.onlyRaw}</span></div>${ds.hasDeadOurs() ? `<div class="row"><span class="k">submission (D + filtered L&amp;H) vs GT, ${ag.nd} frames</span><span class="v">${(100 * ag.sub).toFixed(1)} %</span></div><div class="row"><span class="k">our dead ball: recall / precision</span><span class="v">${(100 * ag.deadRecall).toFixed(0)} % / ${(100 * ag.deadPrecision).toFixed(0)} %</span></div>` : ''}<div class="row"><span class="k">filter moved ball &gt; 2 m</span><span class="v">${ds.meta.ballq.moved.length} places</span></div>${q.white_team ? `<div class="row"><span class="k">light kit</span><span class="v">${ds.teamName(q.white_team)}</span></div>` : ''}${q.config_hash ? `<div class="row"><span class="k">ballq config</span><span class="v mono">${q.config_hash}</span></div>` : ''}`;
  }
  if (!store.layerB.points.length && !store.layerB.segments.length) { el.innerHTML = html + '<span class="hint">import a second annotator\'s JSON as layer B to compare</span>'; return; }
  const ag = store.agreement(12);
  el.innerHTML = html + `<div class="row"><span class="k">team agreement</span><span class="v">${(100 * ag.team).toFixed(1)} %</span></div><div class="row"><span class="k">state agreement</span><span class="v">${(100 * ag.state).toFixed(1)} %</span></div><div class="row"><span class="k">events matched ±0.5 s</span><span class="v">${ag.events.matched} · only A ${ag.events.onlyA} · only B ${ag.events.onlyB}</span></div>`;
}

// ---------------------------------------------------------------- palette, layers, legend, help
function buildPalette() {
  for (const [cat, id] of [['state', '#palState'], ['event', '#palEvent'], ['data', '#palData']]) {
    const el = $(id); el.innerHTML = '';
    for (const [type, key, label, color] of SCHEMA[cat].items) {
      const d = document.createElement('div'); d.className = 'tool'; d.dataset.key = key; d.innerHTML = `<span class="sw" style="background:${color}"></span>${label}<kbd>${key}</kbd>`;
      d.onclick = () => { if (cat === 'state') onKey({ key, target: document.body, preventDefault() {} }); else if (store.pending && store.pending.type === type) closePending(); else arm(key); };
      el.appendChild(d);
    }
  }
}
function buildLayers() {
  const L = $('#layers'); L.innerHTML = '';
  const pr = document.createElement('div'); pr.className = 'presets';
  for (const name in LAYER_PRESETS) { const b = document.createElement('button'); b.textContent = name; b.onclick = () => { const set = LAYER_PRESETS[name]; for (const [k, , on] of LAYERS) layers[k] = set === null ? on : set.includes(k); L.querySelectorAll('input').forEach(i => i.checked = layers[i.dataset.k]); draw(); }; pr.appendChild(b); }
  L.appendChild(pr);
  for (const [k, name, on] of LAYERS) { layers[k] = on; const el = document.createElement('label'); el.className = 'chk'; el.innerHTML = `<input type="checkbox" data-k="${k}" ${on ? 'checked' : ''}> ${name}`; el.querySelector('input').onchange = e => { layers[k] = e.target.checked; draw(); }; L.appendChild(el); }
}
const RESTART_GLYPHS = { 'Throw-In': ['diamond', col('throw')], 'Corner': ['circle', col('corner')], 'Goal Kick': ['square', col('goalkick')], 'Free Kick': ['tri_down', col('freekick')], 'Kick-Off': ['star', col('kickoff')] };
function buildLegend() {
  const el = $('#legend'); el.innerHTML = '';
  const items = [['tri_up', col('good'), 'GT dead → alive', 'RESTART'], ...Object.entries(RESTART_GLYPHS).map(([n, [g, c]]) => [g, c, n.toLowerCase(), 'RESTART']), ['pin', '#fff', 'bookmark', 'BOOKMARK'],
    [null, col('kick'), 'kick', 'KICK'], [null, col('lost'), 'ball lost', 'BALL LOST'], [null, col('jump'), 'teleport', 'PLAYER JUMP'], [null, col('churn'), 'vanish / appear', 'VANISH/APPEAR'], [null, col('bolt'), `› Bolt ${ds.meta.bolt} m/s`, '› BOLT'], [null, TEAM.H, ds.teamName('H'), 'GT'], [null, TEAM.A, ds.teamName('A'), 'GT'], [null, TEAM.D, 'dead ball', 'GT'],
    // the filtered ball is drawn on the video whenever the export carries it, so its entries do not depend on a row
    ['disc', 'rgba(255,255,255,.85)', 'ball as detected', '*ballq'], ['disc', '#8fd3ff', 'ball filtered (mean)', '*ballq'], ['dash', '#8fd3ff', '1 sigma ellipse, dashed link to detector', '*ballq'],
    ['ramp', 'rgb(8,34,110)', 'probability map: darker = more likely', '*grid'],
    [null, TEAM.D, 'DEAD OURS: our dead-ball detector', 'DEAD OURS'], [null, TEAM.H, 'SUB ≠ GT: GT says ' + ds.teamName('H'), 'SUB ≠ GT'], [null, TEAM.A, 'SUB ≠ GT: GT says ' + ds.teamName('A'), 'SUB ≠ GT'], [null, TEAM.D, 'SUB ≠ GT: GT says dead', 'SUB ≠ GT'],
    [null, col('good'), 'DISAGREE: only filtered L&H matches GT', 'DISAGREE'], [null, col('bad'), 'DISAGREE: only raw L&H matches GT', 'DISAGREE'], [null, '#787878', 'DISAGREE: GT has no say', 'DISAGREE'],
    [null, col('warn'), 'BALL MOVED: metres detector to filtered', 'BALL MOVED'], [null, '#8fd3ff', 'BALL SD: spread in metres', 'BALL SD'], [null, col('good'), 'TRUST: how little the detector was believed', 'TRUST']];
  for (const [k, c, t, row] of items) {
    if (row === '*ballq') { if (!ds.hasBallq()) continue; } else if (row === '*grid') { if (!ds.hasGrid()) continue; } else if (row && !rowVisible(row)) continue;
    const s = document.createElement('span'); const cv = document.createElement('canvas'); cv.width = cv.height = 24; const g = cv.getContext('2d'); g.scale(2, 2);
    if (k === 'disc') { g.fillStyle = c; g.beginPath(); g.arc(6, 6, 4, 0, 2 * Math.PI); g.fill(); g.strokeStyle = '#000'; g.lineWidth = .8; g.stroke(); }
    else if (k === 'dash') { g.strokeStyle = c; g.setLineDash([2, 2]); g.beginPath(); g.ellipse(6, 6, 5, 3.5, 0, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]); }
    else if (k === 'ramp') { for (let i = 0; i < 4; i++) { g.fillStyle = c; g.globalAlpha = 0.15 + 0.25 * i; g.fillRect(i * 3, 3, 3, 6); } g.globalAlpha = 1; }
    else if (k) glyph(g, k, 6, 6, 4.5, c); else { g.fillStyle = c; g.fillRect(2, 3, 8, 6); }
    s.appendChild(cv); s.append(t); el.appendChild(s);
  }
}
function setupHelp() {
  const rows = [['space', 'play / pause'], ['← →', 'one frame'], ['⇧ + ← →', 'one second'], [', .', 'previous / next event (type in the timeline bar)'], ['[ ]', 'slower / faster'], ['Home End', 'start / end'],
    ['Enter', 'candidate done, go to next'], ['Backspace', 'candidate skipped'], ['!', 'candidate flagged'], ['H A', 'segment: team from here'], ['1 2 3', 'segment state: controlled / flight / loose'], ['D or 4', 'segment: dead ball'], ['X or 0', 'segment: clear from here'],
    ['P S T I C J F O K', 'event at this frame'], ['N M G W V Y U R Q', 'data error: opens an interval; same key or Enter closes; Esc discards'], ['arrows while placing', 'nudge 10 cm (⇧ 50 cm)'], ['Z', 'magnifier'], ['wheel on video · - / =', 'zoom video at cursor (drag, middle or right drag pans)'], ['L', 'loop the last 2 s'], ['B', 'bookmark with the note'],
    ['Ctrl+Z', 'undo'], ['Esc', 'cancel tool / reset zooms and pins / close help'], ['?', 'this help'], ['click player', 'pin (⇧-click two: distance)'], ['timeline', 'click = seek · drag = zoom · wheel = zoom at cursor'], ['slot row label', 'pin that slot']];
  $('#helpBody').innerHTML = rows.map(([k, d]) => `<div class="row"><span class="k"><kbd>${k}</kbd></span><span class="v">${d}</span></div>`).join('');
  $('#bHelp').onclick = toggleHelp; $('#help').onclick = e => { if (e.target.id === 'help') toggleHelp(); };
}
function toggleHelp() { const h = $('#help'); h.style.display = h.style.display === 'block' ? 'none' : 'block'; }

// ---------------------------------------------------------------- pitch overlay
const PITCH_LINES = (() => {
  const L = [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], [[0.5, 0], [0.5, 1]]];
  const c = []; for (let i = 0; i <= 48; i++) { const t = i / 48 * 2 * Math.PI; c.push([0.5 + 9.15 / 105.3 * Math.cos(t), 0.5 + 9.15 / 68 * Math.sin(t)]); } L.push(c);
  for (const [dep, wid] of [[16.5, 40.32], [5.5, 18.32]]) { const d = dep / 105.3, hw = wid / 2 / 68; L.push([[0, .5 - hw], [d, .5 - hw], [d, .5 + hw], [0, .5 + hw]]); L.push([[1, .5 - hw], [1 - d, .5 - hw], [1 - d, .5 + hw], [1, .5 + hw]]); }
  return L;
})();
function groundCircle(Hm, cx, cy, rm, n = 36) { const o = []; for (let i = 0; i < n; i++) { const t = i / n * 2 * Math.PI; o.push(ds.proj(Hm, cx + rm / ds.pitch[0] * Math.cos(t), cy + rm / ds.pitch[1] * Math.sin(t))); } return o; }
function poly(ctx, pts, close) { ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); if (close) ctx.closePath(); }
let hitList = [];
function draw() {
  if (!ds || !P) return;
  const ov = $('#ov'), ctx = ov.getContext('2d'); ctx.clearRect(0, 0, ov.width, ov.height);
  const r = cur, Hm = ds.H(r), b = ds.ball(r), PL = ds.players(r), tp = +$('#rad').value, holder = P.lhPid[r];
  hitList = []; ctx.lineWidth = 1.2; ctx.font = '500 13px Inter';
  if (!Hm) {
    ctx.fillStyle = 'rgba(240,60,60,.9)'; ctx.font = '500 14px Inter';
    ctx.fillText('no homography on this frame', 16, ov.height - 16);
    if (layers.referee) drawReferee(ctx, r);
    drawPossCard(r, b, tp, 0, 0); return;
  }
  if (layers.pitch) { ctx.strokeStyle = 'rgba(255,255,255,.55)'; for (const L of PITCH_LINES) { poly(ctx, L.map(p => ds.proj(Hm, p[0], p[1]))); ctx.stroke(); } }
  if (layers.regions && PL.length) { const gw = 30, gh = 20; ctx.globalAlpha = .18; for (let i = 0; i < gw; i++) for (let j = 0; j < gh; j++) { const cx = (i + .5) / gw, cy = (j + .5) / gh; let bh = 1e9, ba = 1e9; for (const p of PL) { const d = ds.metres({ x: cx, y: cy }, p); if (ds.sideOf(p.k) === 'H') bh = Math.min(bh, d); else if (ds.sideOf(p.k) === 'A') ba = Math.min(ba, d); } if (bh === ba) continue; ctx.fillStyle = bh < ba ? TEAM.H : TEAM.A; poly(ctx, [[i / gw, j / gh], [(i + 1) / gw, j / gh], [(i + 1) / gw, (j + 1) / gh], [i / gw, (j + 1) / gh]].map(p => ds.proj(Hm, p[0], p[1])), true); ctx.fill(); } ctx.globalAlpha = 1; }
  const hd = ds.attackDir(ds.start + r);
  if (layers.arrows && hd) for (const [c, x0, x1] of [[TEAM.H, .34, .44], [TEAM.A, .66, .56]]) { let a = x0, bb = x1; if (hd < 0) { a = 1 - a; bb = 1 - bb; } const p0 = ds.proj(Hm, a, -.035), p1 = ds.proj(Hm, bb, -.035); ctx.strokeStyle = c; ctx.lineWidth = 2; poly(ctx, [p0, p1]); ctx.stroke(); const ang = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]); poly(ctx, [[p1[0] - 9 * Math.cos(ang - .5), p1[1] - 9 * Math.sin(ang - .5)], p1, [p1[0] - 9 * Math.cos(ang + .5), p1[1] - 9 * Math.sin(ang + .5)]]); ctx.stroke(); ctx.lineWidth = 1.2; }
  if (b && layers.trail) { const n = Math.round(1.5 * ds.fps); ctx.strokeStyle = '#fff'; for (let i = Math.max(1, r - n); i <= r; i++) { const a = ds.ball(i - 1), c = ds.ball(i); if (!a || !c) continue; ctx.globalAlpha = .15 + .6 * (i - (r - n)) / n; poly(ctx, [ds.proj(Hm, a.x, a.y), ds.proj(Hm, c.x, c.y)]); ctx.stroke(); } ctx.globalAlpha = 1; }
  if (b && layers.future) { ctx.fillStyle = '#fff'; for (let i = r + 3; i <= Math.min(ds.n - 1, r + ds.fps); i += 3) { const c = ds.ball(i); if (!c) continue; const p = ds.proj(Hm, c.x, c.y); ctx.beginPath(); ctx.arc(p[0], p[1], 1.4, 0, 7); ctx.fill(); } }
  if (layers.events) {
    for (const kr of E.kick) { const age = (r - kr) / ds.fps; if (age >= 0 && age < .4) { const kb = ds.ball(kr); if (kb) { ctx.strokeStyle = col('kick'); ctx.globalAlpha = 1 - age / .4; poly(ctx, groundCircle(Hm, kb.x, kb.y, .6 + 3 * age), true); ctx.stroke(); ctx.globalAlpha = 1; } } }
    for (const [jr, k] of ds.meta.events.pjump) { const age = (r - jr) / ds.fps; if (age >= 0 && age < .8 && jr > 0) { const a = ds.slot(jr - 1, k), c = ds.slot(jr, k); if (a && c) { ctx.strokeStyle = col('jump'); ctx.lineWidth = 2; ctx.globalAlpha = 1 - age / .8; poly(ctx, [ds.proj(Hm, a.x, a.y), ds.proj(Hm, c.x, c.y)]); ctx.stroke(); ctx.globalAlpha = 1; ctx.lineWidth = 1.2; } } }
    for (const [jr] of ds.meta.events.bjump) { const age = (r - jr) / ds.fps; if (age >= 0 && age < .8 && jr > 0) { const a = ds.ball(jr - 1), c = ds.ball(jr); if (a && c) { ctx.strokeStyle = col('jump'); ctx.lineWidth = 2; ctx.globalAlpha = 1 - age / .8; poly(ctx, [ds.proj(Hm, a.x, a.y), ds.proj(Hm, c.x, c.y)]); ctx.stroke(); ctx.globalAlpha = 1; ctx.lineWidth = 1.2; } } }
    for (const [vr, k] of ds.meta.events.vanish) { const age = (r - vr) / ds.fps; if (age >= 0 && age < 1 && vr > 0) { const a = ds.slot(vr - 1, k); if (a) { const p = ds.proj(Hm, a.x, a.y); ctx.strokeStyle = col('churn'); ctx.globalAlpha = 1 - age; ctx.beginPath(); ctx.arc(p[0], p[1], 8, 0, 7); ctx.stroke(); ctx.fillStyle = col('churn'); ctx.fillText('lost #' + ds.numOf(k), p[0] + 11, p[1] - 10); ctx.globalAlpha = 1; } } }
    for (const [ar, k] of ds.meta.events.appear) { const age = (r - ar) / ds.fps; if (age >= 0 && age < .6) { const a = ds.slot(ar, k); if (a) { ctx.strokeStyle = col('churn'); ctx.globalAlpha = 1 - age / .6; poly(ctx, groundCircle(Hm, a.x, a.y, .5 + 1.5 * age), true); ctx.stroke(); ctx.globalAlpha = 1; } } }
  }
  let inH = 0, inA = 0; const nearest = { H: null, A: null };
  if (b) {
    for (const p of PL) { const d = ds.metres(b, p), s = ds.sideOf(p.k); if (s === 'H' || s === 'A') { if (d <= tp) s === 'H' ? inH++ : inA++; if (!nearest[s] || d < nearest[s].d) nearest[s] = { d, k: p.k, p }; } }
    if (layers.ring) { poly(ctx, groundCircle(Hm, b.x, b.y, tp), true); const tint = inH && !inA ? TEAM.H : inA && !inH ? TEAM.A : null; if (tint) { ctx.fillStyle = tint; ctx.globalAlpha = .16; ctx.fill(); ctx.globalAlpha = 1; } ctx.strokeStyle = col('ring'); ctx.stroke(); }
  }
  if (layers.hull) for (const s of ['H', 'A']) { const pts = PL.filter(p => ds.sideOf(p.k) === s).map(p => ds.proj(Hm, p.x, p.y)); if (pts.length >= 3) { const hull = convexHull(pts); ctx.fillStyle = TEAM[s]; ctx.globalAlpha = .08; poly(ctx, hull, true); ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = TEAM[s]; ctx.stroke(); } }
  if (layers.players) for (const p of PL) {
    const [x, y] = ds.proj(Hm, p.x, p.y), s = ds.sideOf(p.k), c = TEAM[s] || '#888';
    const isHolder = p.k === holder, isPin = pinned.includes(p.k), isHov = hover === p.k;
    if (isHolder && layers.holder) { ctx.fillStyle = c; ctx.globalAlpha = .38; poly(ctx, groundCircle(Hm, p.x, p.y, .6), true); ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = c; ctx.stroke(); }
    if (isPin) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; poly(ctx, groundCircle(Hm, p.x, p.y, .45), true); ctx.stroke(); ctx.lineWidth = 1.2; const n = Math.round(3 * ds.fps); for (let i = Math.max(1, r - n); i <= r; i++) { const a = ds.slot(i - 1, p.k), d = ds.slot(i, p.k); if (!a || !d) continue; ctx.strokeStyle = c; ctx.globalAlpha = .2 + .6 * (i - (r - n)) / n; poly(ctx, [ds.proj(Hm, a.x, a.y), ds.proj(Hm, d.x, d.y)]); ctx.stroke(); } ctx.globalAlpha = 1; }
    ctx.beginPath(); ctx.arc(x, y, isHov ? 10 : 8, 0, 7); ctx.strokeStyle = '#000'; ctx.lineWidth = 3; ctx.stroke(); ctx.strokeStyle = c; ctx.lineWidth = isHov || isPin ? 2 : 1.2; ctx.stroke(); ctx.lineWidth = 1.2;
    if (layers.labels || isHolder || isHov || isPin) { let lab = ds.numOf(p.k); if (isHolder && b) lab += '  ' + ds.metres(b, p).toFixed(1) + ' m'; ctx.font = (isHolder ? '600 ' : '500 ') + '12px Inter'; ctx.fillStyle = '#000'; ctx.fillText(lab, x + 11, y - 9); ctx.fillStyle = c; ctx.fillText(lab, x + 10, y - 10); }
    hitList.push({ x, y, k: p.k });
  }
  if (b && holder >= 0 && layers.opp) { const os = ds.sideOf(holder) === 'H' ? 'A' : 'H', nn = nearest[os]; if (nn) { const o = ds.proj(Hm, nn.p.x, nn.p.y), bp = ds.proj(Hm, b.x, b.y); ctx.strokeStyle = TEAM[os]; ctx.globalAlpha = .7; poly(ctx, [o, bp]); ctx.stroke(); ctx.globalAlpha = 1; ctx.fillStyle = TEAM[os]; ctx.font = '500 11px Inter'; ctx.fillText(nn.d.toFixed(1) + ' m', (o[0] + bp[0]) / 2 + 4, (o[1] + bp[1]) / 2 - 6); } }
  if (pinned.length === 2) { const a = ds.slot(r, pinned[0]), c = ds.slot(r, pinned[1]); if (a && c) { const pa = ds.proj(Hm, a.x, a.y), pc = ds.proj(Hm, c.x, c.y); ctx.setLineDash([4, 4]); ctx.strokeStyle = '#fff'; poly(ctx, [pa, pc]); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = '#fff'; ctx.font = '500 12px Inter'; ctx.fillText(ds.metres(a, c).toFixed(1) + ' m', (pa[0] + pc[0]) / 2 + 4, (pa[1] + pc[1]) / 2 - 6); } }
  if (layers.heat && ds.hasGrid()) {
    // each cell is a quadrilateral on the ground; opacity follows the square root of its mass so the
    // tail of the distribution stays visible next to the peak, and neighbouring frames are prefetched
    const cells = ds.gridCells(r);
    if (cells && cells.n) {
      ctx.fillStyle = 'rgb(8, 34, 110)';
      for (let k = 0; k < cells.n; k++) {
        const [x0, y0, x1, y1] = ds.cellBox(cells.i[k], cells.j[k]);
        ctx.globalAlpha = 0.08 + 0.72 * Math.sqrt(cells.p[k] / cells.max);
        poly(ctx, [ds.proj(Hm, x0, y0), ds.proj(Hm, x1, y0), ds.proj(Hm, x1, y1), ds.proj(Hm, x0, y1)], true); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    for (let k = 1; k <= 8; k++) ds.gridCells(r + k);
  }
  const bq = ds.ballq(r);
  if (bq && layers.ballq) {
    // one sigma ellipse on the ground, then the filtered mean as a cyan disc
    const pts = []; for (let i = 0; i < 36; i++) { const t = i / 36 * 2 * Math.PI; pts.push(ds.proj(Hm, bq.x + bq.sdx * Math.cos(t), bq.y + bq.sdy * Math.sin(t))); }
    ctx.strokeStyle = 'rgba(143,211,255,.8)'; ctx.setLineDash([4, 3]); poly(ctx, pts, true); ctx.stroke(); ctx.setLineDash([]);
    poly(ctx, groundCircle(Hm, bq.x, bq.y, ds.meta.ball_radius, 16), true); ctx.fillStyle = 'rgba(143,211,255,.85)'; ctx.fill(); ctx.strokeStyle = '#003'; ctx.lineWidth = 1; ctx.stroke(); ctx.lineWidth = 1.2;
    if (b && layers.ballqlink && ds.metres(b, bq) > 0.3) { const p1 = ds.proj(Hm, b.x, b.y), p2 = ds.proj(Hm, bq.x, bq.y); ctx.strokeStyle = 'rgba(143,211,255,.7)'; ctx.setLineDash([2, 3]); poly(ctx, [p1, p2]); ctx.stroke(); ctx.setLineDash([]); }
  }
  if (b && layers.ball) { poly(ctx, groundCircle(Hm, b.x, b.y, ds.meta.ball_radius, 16), true); ctx.fillStyle = 'rgba(255,255,255,.6)'; ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke(); ctx.lineWidth = 1.2; const pb = ds.ball(r - 1); if (pb && b.sf > 1) { const c = ds.proj(Hm, b.x, b.y), e = ds.proj(Hm, b.x + (b.x - pb.x) * 10, b.y + (b.y - pb.y) * 10); ctx.strokeStyle = '#fff'; poly(ctx, [c, e]); ctx.stroke(); } }
  if (layers.referee) drawReferee(ctx, r);
  if (layers.ann) { drawAnn(ctx, Hm, r, 'A'); drawAnn(ctx, Hm, r, 'B'); drawPending(ctx, Hm); }
  drawPossCard(r, b, tp, inH, inA);
}
function drawReferee(ctx, r) {
  if (!E || !E.refereeByFrame) return;
  const dets = E.refereeByFrame.get(r) || [];
  // HUD so a missing box is diagnosable (wrong frame vs missing data)
  ctx.save();
  ctx.font = '600 13px Inter';
  ctx.fillStyle = 'rgba(0,0,0,.55)';
  ctx.fillRect(8, 8, 280, dets.length ? 28 : 46);
  ctx.fillStyle = '#ffdc00';
  if (!E.refereeFrames.length) {
    ctx.fillText('AR: no referee JSON loaded', 16, 28);
  } else if (!dets.length) {
    const nearest = E.refereeFrames.reduce((best, f) => best === null || Math.abs(f - r) < Math.abs(best - r) ? f : best, null);
    ctx.fillText(`AR: no box on this frame (${ds.start + r})`, 16, 28);
    if (nearest !== null) ctx.fillText(`nearest @ ${ds.start + nearest}  (step: AR flags)`, 16, 46);
  } else {
    ctx.fillText(`AR: ${dets.length} box(es) on frame ${ds.start + r}`, 16, 28);
  }
  ctx.restore();

  for (const d of dets) {
    const c = (d.hex && d.hex !== '#505050' && d.hex !== '#787878') ? d.hex : '#ffdc00';
    const bbox = d.bbox_xyxy;
    if (bbox) {
      const [x0, y0, x1, y1] = bbox.map(Number);
      const w = x1 - x0, h = y1 - y0;
      // dark halo then bright box — visible on grass and stands
      ctx.strokeStyle = '#000'; ctx.lineWidth = 5; ctx.strokeRect(x0, y0, w, h);
      ctx.strokeStyle = c; ctx.lineWidth = 3; ctx.strokeRect(x0, y0, w, h);
      ctx.font = '700 14px Inter';
      const lab = `${d.official_id || 'AR'} · ${d.signal || '?'}`;
      ctx.lineWidth = 3; ctx.strokeStyle = '#000'; ctx.strokeText(lab, x0 + 2, Math.max(18, y0 - 8));
      ctx.fillStyle = c; ctx.fillText(lab, x0 + 2, Math.max(18, y0 - 8));
      const mx = (x0 + x1) / 2, my = y0 + 0.4 * (y1 - y0);
      if (d.flag_centroid) {
        const [fx, fy] = d.flag_centroid.map(Number);
        ctx.strokeStyle = '#000'; ctx.lineWidth = 4;
        ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(fx, fy); ctx.stroke();
        ctx.strokeStyle = c; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(fx, fy); ctx.stroke();
        ctx.beginPath(); ctx.arc(fx, fy, 6, 0, 7); ctx.fillStyle = c; ctx.fill();
        ctx.strokeStyle = '#000'; ctx.lineWidth = 2; ctx.stroke();
      }
    } else if (d.flag_centroid) {
      const [fx, fy] = d.flag_centroid.map(Number);
      ctx.fillStyle = c; ctx.beginPath(); ctx.arc(fx, fy, 6, 0, 7); ctx.fill();
      glyph(ctx, d.glyph || 'tri_up', fx, fy - 14, 5, c);
    }
  }
}
function convexHull(pts) { pts = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]); const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); const lo = [], up = []; for (const p of pts) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); } for (const p of pts.reverse()) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); } return lo.slice(0, -1).concat(up.slice(0, -1)); }
function annAnchor(a, Hm, r) { if (a.x !== undefined) return ds.proj(Hm, a.x, a.y); if (a.slots) { const s = ds.slot(r, a.slots[0]); if (s) return ds.proj(Hm, s.x, s.y); } if (a.bx !== undefined) return ds.proj(Hm, a.bx, a.by); return null; }
function drawAnn(ctx, Hm, r, layer) {
  const hollow = layer === 'B';
  for (const a of store.near(r, 0.6 * ds.fps, layer)) {
    const inside = a.f <= r && r <= (a.f2 ?? a.f); const al = inside ? 1 : 0.35; const c = annColor(a);
    const p = annAnchor(a, Hm, r); if (!p) continue;
    ctx.globalAlpha = al; ctx.strokeStyle = c; ctx.fillStyle = c; ctx.lineWidth = hollow ? 1 : 1.5; if (hollow) ctx.setLineDash([2, 3]);
    if (a.type === 'ball_pos') { ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 7); ctx.stroke(); ctx.beginPath(); ctx.moveTo(p[0] - 10, p[1]); ctx.lineTo(p[0] + 10, p[1]); ctx.moveTo(p[0], p[1] - 10); ctx.lineTo(p[0], p[1] + 10); ctx.stroke(); const b = ds.ball(r); if (b && inside) { const q = ds.proj(Hm, b.x, b.y); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(q[0], q[1]); ctx.lineTo(p[0], p[1]); ctx.stroke(); } }
    else if (a.type === 'player_missing') { ctx.strokeStyle = TEAM[a.team] || c; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(p[0], p[1], 9, 0, 7); ctx.stroke(); ctx.fillStyle = TEAM[a.team] || c; ctx.fillText('+ ' + (a.team === 'H' ? 'BAR' : 'ATH'), p[0] + 11, p[1] - 10); }
    else if (a.type === 'id_swap' && a.slots && a.slots[1] !== undefined) { const s2 = ds.slot(r, a.slots[1]); if (s2) { const q = ds.proj(Hm, s2.x, s2.y); ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); ctx.fillText('⇄', (p[0] + q[0]) / 2, (p[1] + q[1]) / 2 - 4); } }
    else if (a.cat === 'data') { ctx.beginPath(); ctx.rect(p[0] - 11, p[1] - 11, 22, 22); ctx.stroke(); ctx.fillText(annLabel(a) + (hollow ? ' (B)' : ''), p[0] + 14, p[1] + 20); }
    else if (a.cat === 'event') { glyph(ctx, 'diamond', p[0], p[1] - 18, 6, c, hollow); ctx.fillText(annLabel(a) + (hollow ? ' (B)' : ''), p[0] + 9, p[1] - 20); }
    ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.lineWidth = 1.2;
  }
}
function drawPending(ctx, Hm) {
  const p = store.pending; if (!p) return;
  if (p.x !== undefined) { const q = ds.proj(Hm, p.x, p.y); ctx.strokeStyle = '#fff'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(q[0], q[1], 9, 0, 7); ctx.stroke(); ctx.beginPath(); ctx.moveTo(q[0] - 14, q[1]); ctx.lineTo(q[0] + 14, q[1]); ctx.moveTo(q[0], q[1] - 14); ctx.lineTo(q[0], q[1] + 14); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = '#fff'; ctx.font = '500 11px Inter'; ctx.fillText(`${TYPE[p.type].label} (open)`, q[0] + 16, q[1] - 12); }
  else if (pendingClicks.length && pendingClicks[0].x !== undefined) { const q = ds.proj(Hm, pendingClicks[0].x, pendingClicks[0].y); ctx.strokeStyle = '#fff'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(q[0], q[1], 9, 0, 7); ctx.stroke(); ctx.setLineDash([]); }
}
function drawPossCard(r, b, tp, inH, inA) {
  const g = ds.gt(r), el = $('#possCards'); let html = '';
  const name = { H: ds.teamName('H') + ' · home', A: ds.teamName('A') + ' · away', D: 'dead ball', X: 'no label' };
  if (g !== null) { let s = r; while (s > 0 && ds.gt(s - 1) === g) s--; html += `<div class="card" style="border-top:3px solid ${TEAM[g]}"><div class="lbl">ground truth</div><div class="big" style="color:${TEAM[g]}">${g}</div><div class="name">${name[g]}</div><div class="sub">held ${((r - s) / ds.fps).toFixed(1)} s</div></div>`; }
  else html += `<div class="card"><div class="lbl">ground truth</div><div class="sub">none on the broadcast timeline</div></div>`;
  if (ds.meta.pred) { const p = ds.pred(r), ok = p === g; html += `<div class="card" style="border-top:3px solid ${ok ? col('good') : col('bad')}"><div class="lbl">prediction</div><div class="big" style="color:${ok ? col('good') : col('bad')}">${p}</div><div class="sub">${ok ? 'agrees' : 'disagrees'}</div></div>`; }
  html += `<div class="card"><div class="lbl">heuristics</div><div class="row"><span class="k">naive</span><span class="v" style="color:${TEAM[P.naive[r]]}">${P.naive[r] || '-'}</span></div><div class="row"><span class="k">L&amp;H raw ball</span><span class="v" style="color:${TEAM[P.lh[r]]}">${P.lh[r] || '-'}</span></div>${ds.hasBallq() ? `<div class="row"><span class="k">L&amp;H filtered ball</span><span class="v" style="color:${TEAM[ds.lhPost(r)]}">${ds.lhPost(r) === 'X' ? '-' : ds.lhPost(r)}${ds.ballq(r) ? ` <span class="hint">p ${ds.ballq(r).hp.toFixed(2)}</span>` : ''}</span></div>` : ''}<div class="row"><span class="k">within ${tp} m</span><span class="v"><span style="color:${TEAM.H}">H ${inH}</span> <span style="color:${TEAM.A}">A ${inA}</span></span></div>${ds.hasDeadOurs() ? `<div class="row"><span class="k">submission</span><span class="v" style="color:${TEAM[ds.submission(r)]}">${ds.submission(r) || '-'}${ds.deadOurs(r) ? ` <span class="hint">our dead ball</span>` : ''}</span></div>` : ''}${AG ? `<div class="row"><span class="k">vs GT</span><span class="v">raw ${(100 * AG.raw).toFixed(1)} %</span></div><div class="row"><span class="k">(${AG.n} frames)</span><span class="v" style="color:${AG.post >= AG.raw ? col('good') : col('bad')}">filtered ${(100 * AG.post).toFixed(1)} %</span></div>${ds.hasDeadOurs() ? `<div class="row"><span class="k">submission (${AG.nd})</span><span class="v">${(100 * AG.sub).toFixed(1)} %</span></div>` : ''}` : ''}</div>`;
  const seg = store.segmentAt(r); if (seg.state !== 'clear') { const dl = store.derivedLabel(r); const it = SCHEMA.state.items.find(i => i[0] === seg.state); html += `<div class="card" style="border-top:3px solid ${TEAM[dl]}"><div class="lbl">my reading</div><div class="big" style="color:${TEAM[dl]}">${dl}</div><div class="sub" style="color:${it[3]}">${it[2]}</div></div>`; }
  el.innerHTML = html;
}
function pitchXY(e) { const ov = $('#ov'), rct = ov.getBoundingClientRect(); return [(e.clientX - rct.left) * ov.width / rct.width, (e.clientY - rct.top) * ov.height / rct.height]; }
function nearestHit(e) { const [x, y] = pitchXY(e); let best = null, bd = 18; for (const h of hitList) { const d = Math.hypot(h.x - x, h.y - y); if (d < bd) { bd = d; best = h; } } return best; }
function onPitchHover(e) {
  if (magnify) drawMagnifier(e);
  const h = nearestHit(e), tip = $('#tip');
  if (!h) { if (hover !== null) { hover = null; draw(); } tip.style.display = 'none'; return; }
  if (hover !== h.k) { hover = h.k; draw(); }
  const p = ds.slot(cur, h.k), b = ds.ball(cur), s = ds.sideOf(h.k), wr = $('#wrap').getBoundingClientRect();
  tip.style.display = 'block'; tip.style.left = (e.clientX - wr.left + 14) + 'px'; tip.style.top = (e.clientY - wr.top + 14) + 'px';
  tip.innerHTML = `<b style="color:${TEAM[s]}">#${ds.numOf(h.k)}</b> ${ds.teamName(s)}<br>speed data ${fmt(p.sd)} · frames ${fmt(p.sf)} m/s${b ? `<br>to ball ${ds.metres(b, p).toFixed(2)} m` : ''}<br>x ${p.x.toFixed(3)} y ${p.y.toFixed(3)}`;
}
function drawMagnifier(e) {
  const m = $('#mag'), v = $('#v'), ov = $('#ov'), wr = $('#wrap').getBoundingClientRect(); const [px, py] = pitchXY(e); const size = 220, zoom = 2.5;
  m.width = size; m.height = size; m.style.left = (e.clientX - wr.left + 20) + 'px'; m.style.top = (e.clientY - wr.top - size - 20) + 'px';
  const g = m.getContext('2d'); const sw = size / zoom; g.drawImage(v, px - sw / 2, py - sw / 2, sw, sw, 0, 0, size, size); g.drawImage(ov, px - sw / 2, py - sw / 2, sw, sw, 0, 0, size, size);
  g.strokeStyle = '#fff'; g.beginPath(); g.moveTo(size / 2 - 8, size / 2); g.lineTo(size / 2 + 8, size / 2); g.moveTo(size / 2, size / 2 - 8); g.lineTo(size / 2, size / 2 + 8); g.stroke();
}
function onPitchClick(e) {
  if (suppressClick) return;
  if (armed || (store.pending && store.pending.x !== undefined)) { toolClick(e); return; }
  const h = nearestHit(e); if (!h) return;
  if (e.shiftKey) { if (!pinned.includes(h.k)) pinned = [...pinned.slice(-1), h.k]; } else pinned = pinned.includes(h.k) ? [] : [h.k];
  renderPinned(); draw(); drawTimeline();
}
function renderPinned() { const el = $('#pinned'); if (!pinned.length) { el.className = 'hint'; el.textContent = 'click a player on the pitch to pin; shift-click two for a distance; click a slot row label to pin that slot'; return; } el.className = ''; el.innerHTML = pinned.map(k => `<div class="row"><span class="k" style="color:${TEAM[ds.sideOf(k)]}">#${ds.numOf(k)}</span><span class="v" id="pin${k}"></span></div>`).join(''); updateInfo(); }

// ---------------------------------------------------------------- side info
function updateInfo() {
  const r = cur, f = ds.start + r, t = f / ds.fps, b = ds.ball(r), PL = ds.players(r);
  $('#clkFrame').textContent = 'frame ' + f; $('#clkTime').textContent = `${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(2).padStart(5, '0')}`;
  const h = ds.halfOf(f); let ht = ''; if (h) { const a = ds.meta.halves[h][0]; const s = (f - a) / ds.fps + (h === 2 ? 2700 : 0); ht = `${h === 1 ? '1st' : '2nd'} half ${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`; } $('#clkHalf').textContent = ht;
  const rows = []; const row = (k, v, c) => rows.push(`<div class="row"><span class="k">${k}</span><span class="v"${c ? ` style="color:${c}"` : ''}>${v}</span></div>`);
  const bq = ds.ballq(r);
  if (ds.hasGrid()) { const cells = ds.gridCells(r); if (cells) row('probability map', cells.n ? `${cells.n} cells above ${(100 * 0.002).toFixed(1)} %, peak ${(100 * cells.max).toFixed(1)} %` : 'no cells for this frame'); }
  if (bq) { const mv = ds.ballMoved(r); row('ball filtered', `${bq.x.toFixed(3)}, ${bq.y.toFixed(3)} · sd ${Math.hypot(bq.sdx, bq.sdy).toFixed(2)} m${Number.isFinite(mv) ? ` · moved ${mv.toFixed(2)} m` : ''}`, mv > 2 ? col('warn') : '#8fd3ff'); row('detector trust', `${(100 * bq.tau).toFixed(0)} %${bq.flags & 2 ? ' · air' : ''}${bq.flags & 4 ? ' · white blend' : ''}${bq.flags & 8 ? ' · homography' : ''}${bq.flags & 32 ? ' · dead' : ''}`, bq.tau < 0.5 ? col('warn') : null); }
  if (b) { row('ball', `${b.x.toFixed(3)}, ${b.y.toFixed(3)}${Number.isFinite(b.z) ? ' · z ' + b.z.toFixed(2) + ' m' : ''}`); row('ball speed frames / data', `${fmt(b.sf)} / ${fmt(b.sd)} m/s`); row('ball accel', `${fmt(b.acc)} m/s²`, E.kick.has(r) ? col('kick') : null); } else row('ball', 'not tracked', col('bad'));
  const nH = PL.filter(p => ds.sideOf(p.k) === 'H').length, nA = PL.filter(p => ds.sideOf(p.k) === 'A').length;
  row('tracked', `<span style="color:${TEAM.H}">H ${nH}</span> <span style="color:${TEAM.A}">A ${nA}</span> · ${PL.length}`, PL.length !== 22 ? col('warn') : null);
  const k = P.lhPid[r]; if (k >= 0 && b) { const hp = ds.slot(r, k); let press = 0; for (const p of PL) if (ds.sideOf(p.k) !== ds.sideOf(k) && ds.metres(hp, p) <= 3) press++; row('L&H holder', `#${ds.numOf(k)} · pressure ${press} opp &lt; 3 m`, TEAM[ds.sideOf(k)]); }
  let fast = null; for (const p of PL) if (Number.isFinite(p.sd) && (!fast || p.sd > fast.sd)) fast = p; if (fast) row('fastest', `#${ds.numOf(fast.k)} ${fast.sd.toFixed(1)} m/s${fast.sd > ds.meta.bolt ? ' › Bolt' : ''}`, fast.sd > ds.meta.bolt ? col('bolt') : null);
  if (ds.meta.hom_err) { const e = ds.meta.hom_err[r]; row('homography err', e < 0 ? 'none' : e.toFixed(2), e < 0 || e > 3 ? col('bad') : e > 1 ? col('warn') : col('good')); }
  if (E.dead) { const D = E.dead; row('dead ball (sync)', !D.covered[r] && !D.mask[r] ? 'no counterpart' : D.mask[r] ? [D.noView[r] && 'no view (B)', D.ballLost[r] && 'ball lost T∩B'].filter(Boolean).join(' + ') : 'alive', D.mask[r] ? TEAM.D : null); }
  const hd = ds.attackDir(f); if (hd) row('attack', `${ds.teamName('H')} ${hd > 0 ? '→' : '←'} · ${ds.teamName('A')} ${hd > 0 ? '←' : '→'}`);
  for (const e of ds.meta.tags.filter(e => e.a <= f && f < e.b)) row(e.restart ? 'restart' : 'tag', e.restart ? `${e.restart[0]} · ${e.team || '-'}` : `${e.code} · ${e.team || '-'}`, e.restart ? rgb(e.restart[1]) : TEAM[e.side] || null);
  const seg = store.segmentAt(r); if (seg.state !== 'clear') row('my reading', `${seg.team || '-'} · ${seg.state}`, TEAM[store.derivedLabel(r)]);
  if (store.layerB.segments.length) { const sb = store.segmentAt(r, false, 'B'); row('annotator B', `${sb.team || '-'} · ${sb.state}`, TEAM[store.derivedLabel(r, 'B')]); }
  $('#fi').innerHTML = rows.join('');
  for (const kk of pinned) { const p = ds.slot(r, kk), el = $('#pin' + kk); if (el) el.innerHTML = p ? `${fmt(p.sd)} m/s${b ? ' · ' + ds.metres(b, p).toFixed(1) + ' m to ball' : ''}` : 'not tracked'; }
  renderQueue();
  for (const d of $('#annList').children) d.classList.toggle('cur', false);
}

// ---------------------------------------------------------------- timeline
function buildRows() {
  ROWS = [];
  const grp = (name, rows, opts = {}) => { if (rows.length || opts.forceHeader) ROWS.push({ grp: name, ...opts }); ROWS.push(...rows); };
  const lab = (name, get, cmap = TEAM3) => ({ name, kind: 'label', get, cmap });
  const mask = (name, get, color) => ({ name, kind: 'mask', get, color: hex(color) });
  const pr = []; if (ds.meta.gt) pr.push(lab('GT', r => ds.gt(r))); if (ds.meta.pred) pr.push(lab('PRED', r => ds.pred(r)));
  pr.push(lab('NAIVE', r => P.naive[r]), lab('L&H', r => P.lh[r]));
  if (ds.hasDeadOurs()) pr.push(lab('DEAD OURS', r => ds.deadOurs(r) ? 'D' : null));
  if (ds.hasBallq()) {
    pr.push(lab('L&H FILTERED', r => ds.lhPost(r) === 'X' ? null : ds.lhPost(r)));
    if (ds.hasDeadOurs()) {
      pr.push(lab('SUBMISSION', r => ds.submission(r)));
      // where the submission label and GT differ, in the colour of what GT says
      pr.push({ name: 'SUB ≠ GT', kind: 'cat', get: r => { const g = ds.gt(r), s = ds.submission(r); return (g === 'H' || g === 'A' || g === 'D') && s !== g ? hex(TEAM[g]) : null; } });
    }
    // where the raw and the filtered heuristic disagree: green when the filtered one matches GT, red when the raw one does, grey when GT has no say
    pr.push({ name: 'DISAGREE', kind: 'cat', get: r => { const a = P.lh[r], b = ds.lhPost(r), g = ds.gt(r); if (!a || !b || b === 'X' || a === b) return null; return g === b ? hex(col('good')) : g === a ? hex(col('bad')) : [120, 120, 120]; } });
  }
  pr.push({ name: 'SEGMENT', kind: 'segment', layer: 'A' }, lab('MINE', r => store.derivedLabel(r)));
  if (store.layerB.segments.length) pr.push({ name: 'SEGMENT B', kind: 'segment', layer: 'B' });
  grp('possession', pr);
  if (E.dead) {
    const D = E.dead;
    grp('dead ball (sync)', [mask('DEAD BALL', r => D.mask[r], TEAM.D), mask('NO VIEW (HOM ∅ ∩ ≠22)', r => D.noView[r], col('dbview')),
      mask('BALL LOST T∩B', r => D.ballLost[r], col('dbball')), mask('NO COUNTERPART', r => !D.covered[r], '#5f656d')]);
  }
  const ev = [{ name: 'RESTART', kind: 'restart' }, mask('KICK', r => E.kick.has(r), col('kick'))];
  if (E.refereeFrames && E.refereeFrames.length) {
    const refColor = (ds.meta.referee && ds.meta.referee.styles && ds.meta.referee.styles.attention_raised)
      ? hex(ds.meta.referee.styles.attention_raised.hex || '#ffdc00') : hex('#ffdc00');
    ev.push({
      name: 'AR FLAGS', kind: 'cat',
      get: r => {
        const dets = E.refereeByFrame.get(r);
        if (!dets || !dets.length) return null;
        const d = dets.find(x => x.signal && x.signal !== 'none') || dets[0];
        const st = (ds.meta.referee && ds.meta.referee.styles && ds.meta.referee.styles[d.signal]) || null;
        if (st && st.hex) return hex(st.hex);
        if (d.hex) return hex(d.hex);
        return refColor;
      },
    });
  }
  ev.push({ name: 'ANN EVENTS', kind: 'ann', cat: 'event', layer: 'A' });
  if (store.layerB.points.length) ev.push({ name: 'ANN B', kind: 'ann', cat: null, layer: 'B' });
  ev.push(mask('BOOKMARK', r => store.bookmarks.some(b => b.f === r), '#ffffff'));
  grp('events', ev);
  const dq = [mask('BALL LOST', r => !ds.ball(r), col('lost')), { name: 'BALL SPEED', kind: 'int', color: hex('#ffffff'), vmax: 30, bad: 40, get: r => { const b = ds.ball(r); return b ? b.sf : NaN; } }];
  if (ds.meta.has_z) dq.push({ name: 'BALL AIR', kind: 'int', color: hex(col('air')), vmax: 3, get: r => { const b = ds.ball(r); return b ? b.z : NaN; } });
  dq.push(mask('BALL JUMP', r => E.bjump.has(r), col('jump')), mask('PLAYER JUMP', r => E.pjump.has(r), col('jump')), mask('DROPOUT <0.5 s', r => ds.meta.dropout[r], '#777777'), mask('VANISH/APPEAR', r => E.churnFrames.includes(r), col('churn')));
  dq.push({ name: 'COUNT ≠ 22', kind: 'cat', get: r => { const c = ds.meta.count[r]; return c > 22 ? hex(col('churn')) : c < 16 ? hex(col('bad')) : c < 20 ? hex(col('warn')) : c < 22 ? [90, 90, 90] : null; } });
  dq.push(mask('› BOLT', r => ds.meta.pmax[r] > ds.meta.bolt, col('bolt')));
  if (ds.meta.hom_err) dq.push({ name: 'HOMOGRAPHY', kind: 'cat', get: r => { const e = ds.meta.hom_err[r]; return e < 0 ? hex(col('bad')) : e < 1 ? hex(col('good')) : e < 3 ? hex(col('warn')) : hex(col('bad')); } });
  dq.push({ name: 'ANN DATA', kind: 'ann', cat: 'data', layer: 'A' });
  grp('data quality', dq);
  if (ds.hasBallq()) {
    const q = r => ds.ballq(r), bit = (r, m) => { const b = q(r); return b && (b.flags & m); };
    grp('ball filter', [
      { name: 'BALL MOVED', kind: 'int', color: hex(col('warn')), vmax: 3, bad: 5, get: r => ds.ballMoved(r) },
      { name: 'BALL SD', kind: 'int', color: hex('#8fd3ff'), vmax: 2, bad: 4, get: r => { const b = q(r); return b ? Math.hypot(b.sdx, b.sdy) : NaN; } },
      { name: 'TRUST', kind: 'int', color: hex(col('good')), vmax: 1, get: r => { const b = q(r); return b ? 1 - b.tau : NaN; } },
      mask('BALL AIR FLAG', r => bit(r, 2), col('air')), mask('WHITE BLEND', r => bit(r, 4), col('lost')), mask('HOM LOW', r => bit(r, 8), col('bad')), mask('RESTART PRIOR', r => bit(r, 16), col('good')),
    ]);
  }
  const runs = slotRuns(ds); runs.team = k => ds.sideOf(k);
  const order = orderSlots(runs, view[0], view[1]);
  grp(`slots ${slotsOpen ? '▾' : '▸'} (${ds.S})`, slotsOpen ? order.map(k => ({ name: `#${ds.numOf(k)}`, kind: 'slot', k, runs: runs[k], jumps: runs.jumps[k] })) : [], { toggle: true, forceHeader: true });
  rowCache.clear();
}
function rowStrip(row, r0, r1, W) {
  const key = [row.name, r0, r1, W, $('#rad').value, store.segments.length, store.points.length, store.bookmarks.length, store.layerB.points.length, pinned.join()].join('|');
  if (rowCache.has(key)) return rowCache.get(key);
  let px;
  if (row.kind === 'label') px = rasterLabel(row.get, r0, r1, W, row.cmap, BG3);
  else if (row.kind === 'mask') px = rasterMask(row.get, r0, r1, W, row.color, BG3);
  else if (row.kind === 'int') px = rasterIntensity(row.get, r0, r1, W, row.vmax, row.color, BG3, row.bad ? { bad: row.bad, badColor: hex(col('bad')) } : {});
  else if (row.kind === 'cat') px = rasterCategorical(row.get, r0, r1, W, BG3);
  else if (row.kind === 'segment') px = rasterCategorical(r => { const s = store.segmentAt(r, false, row.layer); if (s.state === 'clear') return null; if (s.state === 'dead') return TEAM3.D; if (!s.team) return null; const base = TEAM3[s.team]; const sh = STATE_SHADE[s.state] ?? 0; const c = lighten(base, sh); return s.state === 'loose' && (r % 4 < 2) ? BG3 : c; }, r0, r1, W, BG3);
  else if (row.kind === 'ann') px = rasterInterval(store.items(row.layer).filter(a => !row.cat || a.cat === row.cat).map(a => ({ f: a.f, f2: a.f2, color: hex(annColor(a)) })), r0, r1, W, BG3);
  else if (row.kind === 'restart') { px = rasterInterval(ds.meta.tags.filter(e => e.restart).map(e => ({ f: e.a - ds.start, f2: e.b - ds.start, color: e.restart[1].slice().reverse().map(v => Math.round(v * .55)) })), r0, r1, W, BG3); const m = rasterMask(r => r > 0 && ds.gt(r - 1) === 'D' && 'HA'.includes(ds.gt(r) || ''), r0, r1, W, hex(col('good')), BG3); for (let i = 0; i < W * 3; i += 3) if (m[i] !== BG3[0] || m[i + 1] !== BG3[1] || m[i + 2] !== BG3[2]) { px[i] = m[i]; px[i + 1] = m[i + 1]; px[i + 2] = m[i + 2]; } }
  else if (row.kind === 'slot') px = rasterSlot(row.runs, row.jumps, r0, r1, W, pinned.includes(row.k) ? TEAM3[ds.sideOf(row.k)] : TEAM3[ds.sideOf(row.k)].map(v => Math.round(v * .45 + 34 * .55)), hex(col('jump')), BG3);
  const c = document.createElement('canvas'); c.width = W; c.height = 1; const g = c.getContext('2d'); const img = g.createImageData(W, 1);
  for (let i = 0; i < W; i++) { img.data[i * 4] = px[i * 3]; img.data[i * 4 + 1] = px[i * 3 + 1]; img.data[i * 4 + 2] = px[i * 3 + 2]; img.data[i * 4 + 3] = 255; }
  g.putImageData(img, 0, 0); rowCache.set(key, c); if (rowCache.size > 400) rowCache.delete(rowCache.keys().next().value); return c;
}
function rowVisible(name) { return !store.rows || store.rows.includes(name) || name.startsWith('#'); }
function groupVisible(i) { if (ROWS[i].forceHeader) return true; for (let j = i + 1; j < ROWS.length && !ROWS[j].grp; j++) if (rowVisible(ROWS[j].name)) return true; return false; }
let bands = null, bandBin = 0;
// about 400 bins across the export, never finer than 2 s
function ensureBands() { if (bands) return; bandBin = Math.max(ds.fps * 2, Math.round(ds.n / 400)); bands = aggregateBands(ds, E, P, bandBin); }
function setupTimeline() {
  const c = $('#tlc');
  c.addEventListener('mousedown', e => { if (e.offsetY < overviewHeight() + 4) { $('#v').pause(); seek(Math.round(e.offsetX / c.clientWidth * (ds.n - 1))); return; } tlDrag = { x0: e.offsetX, x1: e.offsetX }; });
  window.addEventListener('mousemove', e => { if (!tlDrag) return; const rct = c.getBoundingClientRect(); tlDrag.x1 = e.clientX - rct.left; drawTimeline(); });
  window.addEventListener('mouseup', () => { if (!tlDrag) return; const [a, b] = [tlDrag.x0, tlDrag.x1].sort((x, y) => x - y), W = c.clientWidth; if (b - a < 4) { $('#v').pause(); seek(xToFrame(a, W)); } else { view = [xToFrame(a, W), xToFrame(b, W)]; if (view[1] - view[0] < 10) view[1] = view[0] + 10; buildRows(); } tlDrag = null; drawTimeline(); });
  c.addEventListener('mousemove', e => { const W = c.clientWidth, r = xToFrame(e.offsetX, W), tip = $('#tip2'); tip.style.display = 'block'; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY - 70) + 'px'; const b = ds.ball(r), seg = store.segmentAt(r); tip.innerHTML = `<b class="mono">${ds.start + r}</b> · ${(r / ds.fps).toFixed(1)} s<br>GT <b style="color:${TEAM[ds.gt(r) || 'X']}">${ds.gt(r) || '-'}</b> · naive <b style="color:${TEAM[P.naive[r]]}">${P.naive[r] || '-'}</b> · L&amp;H <b style="color:${TEAM[P.lh[r]]}">${P.lh[r] || '-'}</b>${ds.hasBallq() ? ` · filtered <b style="color:${TEAM[ds.lhPost(r)]}">${ds.lhPost(r) === 'X' ? '-' : ds.lhPost(r)}</b>` : ''}${ds.meta.pred ? ` · pred <b style="color:${TEAM[ds.pred(r)]}">${ds.pred(r)}</b>` : ''}${seg.state !== 'clear' ? `<br>mine ${seg.team || '-'} · ${seg.state}` : ''}<br>ball ${b ? b.sf.toFixed(1) + ' m/s' : 'lost'} · tracked ${ds.meta.count[r]}${E.dead && E.dead.mask[r] ? ` · <b style="color:${TEAM.D}">dead ball (sync)</b>` : ''}`; });
  c.addEventListener('mouseleave', () => $('#tip2').style.display = 'none');
  c.addEventListener('wheel', e => { e.preventDefault(); if (e.shiftKey) { $('#tlbody').scrollTop += e.deltaY || e.deltaX; return; } const W = c.clientWidth, r = xToFrame(e.offsetX, W), span = view[1] - view[0], k = e.deltaY > 0 ? 1.25 : .8, ns = Math.max(10, Math.min(ds.n - 1, span * k)); let a = r - (r - view[0]) * ns / span; a = Math.max(0, Math.min(ds.n - 1 - ns, a)); view = [Math.round(a), Math.round(a + ns)]; buildRows(); drawTimeline(); }, { passive: false });
  new ResizeObserver(drawTimeline).observe($('#tl'));
}
const xToFrame = (x, W) => Math.max(0, Math.min(ds.n - 1, Math.round(view[0] + x / W * (view[1] - view[0]))));
const frameToX = (r, W) => (r - view[0]) / Math.max(1, view[1] - view[0]) * W;
function drawTimeline() {
  if (!ROWS.length || document.body.classList.contains('notl')) return; ensureBands();
  const c = $('#tlc'), W = Math.max(50, c.clientWidth), dpr = window.devicePixelRatio || 1;
  const items = overviewOn().map(n => OVERVIEW_ITEMS.find(i => i.name === n));
  const ovH = items.reduce((s, it) => s + (it.kind === 'strip' ? OVH : BANDH) + 1, 0);
  const TOP = ovH + 6 + FLAG; let h = TOP;
  ROWS.forEach((r, i) => { if (r.grp) { if (groupVisible(i)) h += GG; } else if (rowVisible(r.name)) h += RH + RG; });
  c.width = W * dpr; c.height = h * dpr; c.style.height = h + 'px'; const g = c.getContext('2d'); g.scale(dpr, dpr);
  g.fillStyle = '#17191c'; g.fillRect(0, 0, W, h);
  // overview: strips and bands over the whole export, in the order the user picked
  const labels = $('#rowlabels'); labels.innerHTML = ''; labels.style.height = h + 'px';
  const mkLab = (t, yy, cls, onclick, fs) => { const d = document.createElement('div'); d.textContent = t; d.title = t; d.style.top = yy + 'px'; if (fs) d.style.fontSize = fs + 'px'; if (cls) d.className = cls; if (onclick) { d.onclick = onclick; d.style.cursor = 'pointer'; } labels.appendChild(d); };
  const nb = bands.length, bw = W / nb; let y = 0;
  for (const it of items) {
    if (it.kind === 'strip') { const row = ROWS.find(r => r.name === it.row); const saved = view; view = [0, ds.n - 1]; g.drawImage(rowStrip(row, 0, ds.n - 1, W), 0, 0, W, 1, 0, y, W, OVH); view = saved; mkLab(it.name, y + 1, 'grp', null, Math.min(10, OVH)); y += OVH + 1; continue; }
    for (const [i, bnd] of bands.entries()) {
      if (it.key === 'share') { const d = bnd.shareH - bnd.shareA; if (!d) continue; g.fillStyle = d > 0 ? TEAM.H : TEAM.A; g.globalAlpha = Math.min(1, Math.abs(d) * 1.3); }
      else if (it.count) { if (!bnd[it.key]) continue; g.fillStyle = col(it.color); g.globalAlpha = Math.min(1, 0.5 + bnd[it.key] * 0.25); }
      else { if (!bnd[it.key]) continue; g.fillStyle = col(it.color); g.globalAlpha = Math.min(1, bnd[it.key] * 1.5); }
      g.fillRect(i * bw, y, Math.max(1, Math.ceil(bw)), BANDH); g.globalAlpha = 1;
    }
    mkLab(it.name, y, 'grp', null, Math.min(10, BANDH + 1)); y += BANDH + 1;
  }
  const bandsH = ovH;
  g.fillStyle = 'rgba(255,255,255,.12)'; g.fillRect(view[0] / (ds.n - 1) * W, 0, Math.max(2, (view[1] - view[0]) / (ds.n - 1) * W), bandsH); g.strokeStyle = '#fff'; g.strokeRect(view[0] / (ds.n - 1) * W + .5, .5, Math.max(2, (view[1] - view[0]) / (ds.n - 1) * W), bandsH - 1);
  g.fillStyle = '#fff'; g.fillRect(cur / (ds.n - 1) * W, 0, 1, bandsH);
  // rows
  y = TOP; const yFlag = TOP - FLAG;
  for (let i = 0; i < ROWS.length; i++) {
    const row = ROWS[i];
    if (row.grp) { if (groupVisible(i)) { mkLab(row.grp, y + 1, 'grp', row.toggle ? () => { slotsOpen = !slotsOpen; buildRows(); drawTimeline(); } : null); y += GG; } continue; }
    if (!rowVisible(row.name)) continue;
    g.drawImage(rowStrip(row, view[0], view[1], W), 0, 0, W, 1, 0, y, W, RH);
    mkLab(row.name, y + Math.max(0, (RH - 10) / 2), row.kind === 'slot' && pinned.includes(row.k) ? 'pin' : null, row.kind === 'slot' ? () => { pinned = pinned.includes(row.k) ? [] : [row.k]; renderPinned(); rowCache.clear(); draw(); drawTimeline(); } : null);
    y += RH + RG;
  }
  for (const f of E.flags) { if (f.r < view[0] || f.r > view[1]) continue; const x = frameToX(f.r, W), cc = rgb(f.color); g.strokeStyle = cc; g.globalAlpha = .5; g.beginPath(); g.moveTo(x + .5, yFlag); g.lineTo(x + .5, h); g.stroke(); g.globalAlpha = 1; glyph(g, f.glyph, x, yFlag + 6, 4.5, cc); }
  for (const r of E.restartFrames) if (r >= view[0] && r <= view[1] && !E.flags.some(f => f.r === r)) { const x = frameToX(r, W); g.strokeStyle = col('good'); g.globalAlpha = .5; g.beginPath(); g.moveTo(x + .5, yFlag); g.lineTo(x + .5, h); g.stroke(); g.globalAlpha = 1; glyph(g, 'tri_up', x, yFlag + 6, 4.5, col('good')); }
  for (const b of store.bookmarks) if (b.f >= view[0] && b.f <= view[1]) glyph(g, 'pin', frameToX(b.f, W), yFlag + 6, 4, '#fff');
  for (const a of store.items('A')) if (a.f >= view[0] && a.f <= view[1]) glyph(g, a.cat === 'data' ? 'square' : 'diamond', frameToX(a.f, W), yFlag + 6, 3.5, annColor(a));
  if (store.pending) { const x = frameToX(store.pending.f, W); g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.moveTo(x + 4, yFlag); g.lineTo(x, yFlag); g.lineTo(x, h); g.lineTo(x + 4, h); g.stroke(); g.lineWidth = 1; g.fillStyle = 'rgba(255,255,255,.12)'; g.fillRect(Math.min(x, frameToX(cur, W)), yFlag, Math.abs(frameToX(cur, W) - x), h - yFlag); }
  for (const r of candFrames) if (r >= view[0] && r <= view[1]) { const x = frameToX(r, W); const s = queue.statusOf(r); g.fillStyle = s === 'done' ? 'rgba(90,220,90,.8)' : s === 'flagged' ? 'rgba(230,80,230,.9)' : s === 'skipped' ? 'rgba(150,150,150,.6)' : 'rgba(255,255,255,.5)'; g.fillRect(x, yFlag + FLAG - 3, 1, 3); }
  if (loop) { g.fillStyle = 'rgba(120,220,255,.12)'; g.fillRect(frameToX(loop[0], W), yFlag, frameToX(loop[1], W) - frameToX(loop[0], W), h - yFlag); }
  const cx = frameToX(cur, W); g.strokeStyle = '#fff'; g.lineWidth = 1; g.beginPath(); g.moveTo(cx + .5, yFlag); g.lineTo(cx + .5, h); g.stroke();
  if (tlDrag) { const [a, b] = [tlDrag.x0, tlDrag.x1].sort((p, qq) => p - qq); g.fillStyle = 'rgba(50,170,235,.18)'; g.fillRect(a, yFlag, b - a, h - yFlag); }
  g.fillStyle = '#9aa0a8'; g.font = '11px Inter'; g.textAlign = 'right'; g.fillText(`${ds.start + view[0]} - ${ds.start + view[1]}  (${((view[1] - view[0]) / ds.fps).toFixed(0)} s) · bands ${(bandBin / ds.fps).toFixed(0)} s per bin`, W - 2, bandsH + 11); g.textAlign = 'left';
}
function glyph(g, kind, x, y, r, c, hollow = false) {
  g.fillStyle = c; g.strokeStyle = c; g.beginPath();
  if (kind === 'circle') g.arc(x, y, r, 0, 7);
  else if (kind === 'square') g.rect(x - r, y - r, 2 * r, 2 * r);
  else if (kind === 'diamond') { g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y); g.closePath(); }
  else if (kind === 'tri_up') { g.moveTo(x, y - r); g.lineTo(x + r, y + r); g.lineTo(x - r, y + r); g.closePath(); }
  else if (kind === 'tri_down') { g.moveTo(x, y + r); g.lineTo(x + r, y - r); g.lineTo(x - r, y - r); g.closePath(); }
  else if (kind === 'star') { for (let i = 0; i < 10; i++) { const rr = i % 2 ? r * .5 : r * 1.3, t = -Math.PI / 2 + i * Math.PI / 5; i ? g.lineTo(x + rr * Math.cos(t), y + rr * Math.sin(t)) : g.moveTo(x + rr * Math.cos(t), y + rr * Math.sin(t)); } g.closePath(); }
  else if (kind === 'pin') { g.moveTo(x, y + r); g.lineTo(x - r * .8, y - r * .4); g.lineTo(x + r * .8, y - r * .4); g.closePath(); }
  else { g.moveTo(x - r, y - r); g.lineTo(x + r, y + r); g.moveTo(x + r, y - r); g.lineTo(x - r, y + r); g.stroke(); return; }
  hollow ? g.stroke() : g.fill();
}
// ---------------------------------------------------------------- overview filter (S11)
// which overview items are on, in display order; GT is forced first
function overviewOn() {
  const picked = store.overview || OVERVIEW_DEFAULT;
  const ok = n => { const it = OVERVIEW_ITEMS.find(i => i.name === n); return it && (it.kind !== 'strip' || ROWS.some(r => r.name === it.row)); };
  return ['GT', ...OVERVIEW_ITEMS.map(i => i.name).filter(n => n !== 'GT' && picked.includes(n))].filter(ok);
}
function overviewHeight() { return overviewOn().reduce((s, n) => s + (OVERVIEW_ITEMS.find(i => i.name === n).kind === 'strip' ? OVH : BANDH) + 1, 0); }
function setupOverviewFilter() {
  const pop = $('#ovpop');
  $('#bOverview').onclick = e => { e.stopPropagation(); $('#rowpop').style.display = 'none'; pop.style.display = pop.style.display === 'block' ? 'none' : 'block'; renderOverviewPop(); };
  document.addEventListener('click', e => { if (!pop.contains(e.target)) pop.style.display = 'none'; });
}
function renderOverviewPop() {
  const pop = $('#ovpop'); pop.innerHTML = '<div class="lbl" style="margin-bottom:6px">overview rows, whole export</div>';
  const on = overviewOn();
  for (const it of OVERVIEW_ITEMS) {
    if (it.kind === 'strip' && !ROWS.some(r => r.name === it.row)) continue;
    const l = document.createElement('label'); l.className = 'chk';
    l.innerHTML = `<input type="checkbox" ${on.includes(it.name) ? 'checked' : ''} ${it.name === 'GT' ? 'disabled' : ''}> ${it.name}`;
    l.querySelector('input').onchange = e => { e.stopPropagation(); let next = on.filter(n => n !== it.name); if (e.target.checked) next.push(it.name); store.setOverview(next); drawTimeline(); renderOverviewPop(); };
    l.onclick = e => e.stopPropagation(); pop.appendChild(l);
  }
  const b = document.createElement('button'); b.textContent = 'defaults'; b.style.marginTop = '6px';
  b.onclick = e => { e.stopPropagation(); store.setOverview(null); renderOverviewPop(); drawTimeline(); }; pop.appendChild(b);
}
// ---------------------------------------------------------------- row filter
function setupRowFilter() {
  const pop = $('#rowpop'); $('#bRows').onclick = e => { e.stopPropagation(); $('#ovpop').style.display = 'none'; pop.style.display = pop.style.display === 'block' ? 'none' : 'block'; renderRowPop(); };
  document.addEventListener('click', e => { if (!pop.contains(e.target)) pop.style.display = 'none'; });
}
function renderRowPop() {
  const pop = $('#rowpop'); pop.innerHTML = '<div class="lbl" style="margin-bottom:6px">presets</div><div class="presets"></div>';
  const pr = pop.querySelector('.presets');
  for (const name in ROW_PRESETS) { const b = document.createElement('button'); b.textContent = name; b.onclick = e => { e.stopPropagation(); store.setRows(ROW_PRESETS[name] ? ROW_PRESETS[name].slice() : null); renderRowPop(); buildLegend(); drawTimeline(); }; pr.appendChild(b); }
  for (const row of ROWS) {
    if (row.grp) { if (row.toggle) continue; const g = document.createElement('div'); g.className = 'lbl grp'; g.textContent = row.grp; pop.appendChild(g); continue; }
    if (row.kind === 'slot') continue;
    const l = document.createElement('label'); l.className = 'chk'; l.innerHTML = `<input type="checkbox" ${rowVisible(row.name) ? 'checked' : ''}> ${row.name}`;
    l.querySelector('input').onchange = e => { e.stopPropagation(); const all = ROWS.filter(r => !r.grp && r.kind !== 'slot').map(r => r.name); const cur_ = store.rows ? store.rows.slice() : all.slice(); const i = cur_.indexOf(row.name); if (e.target.checked && i < 0) cur_.push(row.name); if (!e.target.checked && i >= 0) cur_.splice(i, 1); store.setRows(cur_.length === all.length ? null : cur_); buildLegend(); drawTimeline(); };
    l.onclick = e => e.stopPropagation(); pop.appendChild(l);
  }
}
// a small hook so the e2e test can look inside
// the dual page drives both panes through this: one layer set, one radius, one legend
function setLayers(o) { for (const k in o) if (k in layers) layers[k] = !!o[k]; document.querySelectorAll('#layers input').forEach(i => { i.checked = !!layers[i.dataset.k]; }); draw(); }
function setRad(v) { $('#rad').value = v; $('#radV').textContent = (+v).toFixed(1) + ' m'; P = computePossession(ds, E.kick, +v); rowCache.clear(); buildCandidates(); renderAgreement(); draw(); drawTimeline(); }
window.__pv = { ready: false, overviewOn, rowStrip, rows: () => ROWS, LAYERS, LAYER_PRESETS, get layers() { return { ...layers }; }, setLayers, setRad, get rad() { return +$('#rad').value; }, legendHTML: () => $('#legend').innerHTML, buildLegend, get rowCache() { return rowCache; }, redrawTimeline: () => drawTimeline(), rowVisible, get ROWS() { return ROWS; }, get E() { return E; }, video: $('#v'), get ds() { return ds; }, get store() { return store; }, get P() { return P; }, get queue() { return queue; }, get cur() { return cur; }, get view() { return view; }, get pinned() { return pinned; }, get hitList() { return hitList; }, seek, setCandType(t) { candType = t; $('#candType').value = t; buildCandidates(); } };
setupLayout();
load();
