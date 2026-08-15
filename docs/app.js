/*
 * app.js — UI for the in-browser MODI character CNN.
 *
 * Nothing here talks to the network after the initial weight fetch. Every
 * prediction, feature map and occlusion pass is computed locally by cnn.js.
 */

import { ModiCNN } from './cnn.js';

/* ------------------------------------------------------------------ *
 * Constants
 * ------------------------------------------------------------------ */

const SIZE = 32;          // model input is 32x32
const CH = 3;             // ...RGB
const PATCH = 8;          // occlusion patch size, in input pixels
const STRIDE = 4;         // occlusion stride
const GRID = 8;           // -> 8x8 = 64 occlusion positions
const OCCL_GREY = 0.5;    // value the patch is filled with
// One forward pass costs tens of milliseconds, so the occlusion sweep yields
// back to the browser after every single position. That keeps each blocking
// stretch to roughly one pass instead of freezing for seconds on end.
const CHUNK = 1;          // occlusion passes per animation frame

/*
 * Which layers get a tab in the feature explorer. `key` is the real layer name
 * from the manifest; `label` is what we call it in the UI.
 */
const FEATURE_LAYERS = [
  {
    key: '__input__',
    label: 'Input',
    caption:
      'The raw 32×32 image exactly as it enters the network, with pixel ' +
      'values scaled to 0–1. The dataset glyphs are greyscale scans, so the ' +
      'three colour channels carry identical values.',
  },
  {
    key: 'conv2d',
    label: 'Conv 1',
    caption:
      '32 filters applied straight to the pixels. Each unit sees a 3×3 ' +
      'window, so what they can respond to is limited: the presence of ink, and ' +
      'edges at particular orientations. Channels that stay dark are filters ' +
      'that found nothing matching in this glyph, which is normal.',
  },
  {
    key: 'conv2d_1',
    label: 'Conv 2',
    caption:
      '64 filters over the previous layer’s output. Stacking a second 3×3 ' +
      'convolution means each unit now depends on a 5×5 patch of the original ' +
      'image, so responses track short stroke segments, corners and junctions ' +
      'rather than isolated edges.',
  },
  {
    key: 'conv2d_2',
    label: 'Conv 3',
    caption:
      'After 2×2 max pooling the maps are half the resolution, and these 128 ' +
      'channels each summarise roughly a 14×14 region of the input. Individual ' +
      'tiles are no longer readable as pictures — they are closer to a score ' +
      'for “does this learned combination of parts appear around here”.',
  },
  {
    key: 'conv2d_3',
    label: 'Conv 4',
    caption:
      'The final convolution, 64 channels at 10×10, feeding straight into global ' +
      'average pooling — so only the average strength of each channel survives ' +
      'into the classifier, and position is largely discarded. This layer is very ' +
      'sparse: for any single glyph most channels never fire at all, and the ones ' +
      'that do often fire at only a handful of positions. The mostly-black grid ' +
      'is the honest picture, not a rendering fault.',
  },
];

/* Heatmap ramp, mirroring the CSS legend gradient. Cool/dark = unimportant. */
const RAMP = [
  [0.00, 43, 51, 70],
  [0.45, 122, 90, 36],
  [0.75, 213, 154, 48],
  [1.00, 242, 200, 119],
];

/* ------------------------------------------------------------------ *
 * Small helpers
 * ------------------------------------------------------------------ */

const $ = (id) => document.getElementById(id);

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/** Shared 32x32 scratch surface for upscaling glyphs. */
const scratch = makeCanvas(SIZE, SIZE);
const scratchCtx = scratch.getContext('2d', { willReadFrequently: true });

/** Paints uint8 RGB bytes onto the scratch canvas and returns it. */
function scratchFromRGB(bytes) {
  const img = scratchCtx.createImageData(SIZE, SIZE);
  const d = img.data;
  for (let p = 0, s = 0, t = 0; p < SIZE * SIZE; p++, s += CH, t += 4) {
    d[t] = bytes[s];
    d[t + 1] = bytes[s + 1];
    d[t + 2] = bytes[s + 2];
    d[t + 3] = 255;
  }
  scratchCtx.putImageData(img, 0, 0);
  return scratch;
}

/** Same, but from a normalised Float32Array input tensor. */
function scratchFromFloat(input) {
  const img = scratchCtx.createImageData(SIZE, SIZE);
  const d = img.data;
  for (let p = 0, s = 0, t = 0; p < SIZE * SIZE; p++, s += CH, t += 4) {
    d[t] = Math.max(0, Math.min(255, input[s] * 255));
    d[t + 1] = Math.max(0, Math.min(255, input[s + 1] * 255));
    d[t + 2] = Math.max(0, Math.min(255, input[s + 2] * 255));
    d[t + 3] = 255;
  }
  scratchCtx.putImageData(img, 0, 0);
  return scratch;
}

function rampColor(t) {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (x <= RAMP[i][0]) {
      const a = RAMP[i - 1], b = RAMP[i];
      const f = (x - a[0]) / (b[0] - a[0] || 1);
      return [
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
        a[3] + (b[3] - a[3]) * f,
      ];
    }
  }
  const last = RAMP[RAMP.length - 1];
  return [last[1], last[2], last[3]];
}

const CHECK = '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5 6.5 12 13 4"/></svg>';
const CROSS = '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>';

/* ------------------------------------------------------------------ *
 * State
 * ------------------------------------------------------------------ */

const state = {
  model: null,
  sampleIndex: 0,
  source: 'sample',      // 'sample' | 'drawing'
  input: null,           // Float32Array(32*32*3)
  probs: null,
  activations: null,
  layerKey: 'conv2d',
  occlToken: 0,
  occlRunning: false,
};

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */

async function boot() {
  const loader = $('loader');
  const msg = $('loader-msg');
  try {
    state.model = await ModiCNN.load('weights', (m) => { msg.textContent = m; });
  } catch (err) {
    loader.classList.add('loader--error');
    msg.textContent = 'Could not load the model: ' + err.message;
    console.error(err);
    return;
  }

  buildGallery();
  buildTabs();
  selectLayer(state.layerKey);
  buildBars();
  initOcclusion();
  initDrawMode();

  selectSample(0);

  loader.hidden = true;
  $('page').setAttribute('aria-busy', 'false');
}

/* ------------------------------------------------------------------ *
 * 1. Gallery
 * ------------------------------------------------------------------ */

let galleryButtons = [];

function buildGallery() {
  const host = $('gallery');
  const frag = document.createDocumentFragment();
  galleryButtons = [];

  for (let i = 0; i < state.model.sampleCount; i++) {
    const name = state.model.samples[i];
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'glyph';
    btn.setAttribute('role', 'radio');
    btn.setAttribute('aria-checked', 'false');
    btn.setAttribute('aria-label', `Glyph ${name}`);
    btn.tabIndex = -1;
    btn.dataset.index = String(i);

    // Upscale the 32x32 sample with smoothing off, as specified.
    const c = makeCanvas(64, 64);
    const ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(scratchFromRGB(state.model.sampleRGB(i)), 0, 0, 64, 64);

    const label = document.createElement('span');
    label.className = 'glyph__name';
    label.textContent = name;

    btn.append(c, label);
    btn.addEventListener('click', () => selectSample(i));
    frag.appendChild(btn);
    galleryButtons.push(btn);
  }

  host.appendChild(frag);
  host.addEventListener('keydown', onGalleryKeydown);
}

/* Radiogroup keyboard conventions: arrows move and select, Home/End jump. */
function onGalleryKeydown(e) {
  const keys = ['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'Home', 'End'];
  if (!keys.includes(e.key)) return;

  // Work out how many tiles fit per row so Up/Down move a visual row.
  const perRow = Math.max(1, Math.round($('gallery').clientWidth / galleryButtons[0].offsetWidth));
  const n = galleryButtons.length;
  let i = state.sampleIndex;

  if (e.key === 'ArrowRight') i = (i + 1) % n;
  else if (e.key === 'ArrowLeft') i = (i - 1 + n) % n;
  else if (e.key === 'ArrowDown') i = Math.min(n - 1, i + perRow);
  else if (e.key === 'ArrowUp') i = Math.max(0, i - perRow);
  else if (e.key === 'Home') i = 0;
  else if (e.key === 'End') i = n - 1;

  e.preventDefault();
  selectSample(i);
  galleryButtons[i].focus();
}

function selectSample(i) {
  state.sampleIndex = i;
  state.source = 'sample';

  galleryButtons.forEach((b, k) => {
    const on = k === i;
    b.setAttribute('aria-checked', on ? 'true' : 'false');
    b.tabIndex = on ? 0 : -1;
  });

  drawGhost(i);
  setInput(state.model.sampleInput(i));
}

/* ------------------------------------------------------------------ *
 * Central update: new input -> inference -> repaint everything
 * ------------------------------------------------------------------ */

function setInput(input) {
  state.input = input;
  cancelOcclusion();

  const { probs, activations } = state.model.forward(input, { collect: true });
  state.probs = probs;
  state.activations = activations;

  renderInputCanvas();
  renderBars();
  renderFeatureMaps();
  updateSourceLabel();
}

function renderInputCanvas() {
  const canvas = $('input-canvas');
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratchFromFloat(state.input), 0, 0);
}

function updateSourceLabel() {
  const el = $('input-source');
  el.textContent = state.source === 'sample'
    ? `Dataset sample — true label “${state.model.samples[state.sampleIndex]}”`
    : 'Your drawing, downsampled to 32×32';
}

/* ------------------------------------------------------------------ *
 * 2. Prediction bars
 * ------------------------------------------------------------------ */

const barRows = [];

function buildBars() {
  const host = $('bars');
  for (let i = 0; i < 5; i++) {
    const li = document.createElement('li');
    li.className = 'bar';

    const name = document.createElement('span');
    name.className = 'bar__name';

    const track = document.createElement('span');
    track.className = 'bar__track';
    const fill = document.createElement('span');
    fill.className = 'bar__fill';
    track.appendChild(fill);

    const pct = document.createElement('span');
    pct.className = 'bar__pct';

    li.append(name, track, pct);
    host.appendChild(li);
    barRows.push({ li, name, fill, pct });
  }
}

function renderBars() {
  const probs = state.probs;
  const top = state.model.topK(probs, 5);

  top.forEach((classIdx, rank) => {
    const row = barRows[rank];
    const p = probs[classIdx];
    row.name.textContent = state.model.classes[classIdx];
    row.pct.textContent = (p * 100).toFixed(p >= 0.1 ? 1 : 2) + '%';
    row.fill.style.width = Math.max(p * 100, 0.6) + '%';
    row.li.classList.toggle('bar--top', rank === 0);
  });

  renderVerdict(top[0]);
}

function renderVerdict(topIdx) {
  const el = $('verdict');
  el.classList.remove('verdict--ok', 'verdict--miss');

  if (state.source !== 'sample') {
    el.innerHTML = '<span>Freehand input — no ground-truth label to check against.</span>';
    return;
  }

  const predicted = state.model.classes[topIdx];
  const truth = state.model.samples[state.sampleIndex];
  const ok = predicted === truth;
  el.classList.add(ok ? 'verdict--ok' : 'verdict--miss');
  el.innerHTML = ok
    ? `${CHECK}<span>Top prediction matches the labelled class.</span>`
    : `${CROSS}<span>Top prediction is “${predicted}”; the labelled class is “${truth}”.</span>`;
}

/* ------------------------------------------------------------------ *
 * 3. Feature-map explorer
 * ------------------------------------------------------------------ */

const tilePool = [];

function buildTabs() {
  const host = $('feat-tabs');

  FEATURE_LAYERS.forEach((layer, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'tab';
    btn.id = 'tab-' + i;
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', 'false');
    btn.tabIndex = -1;
    btn.dataset.key = layer.key;

    const count = layer.key === '__input__'
      ? CH
      : channelsOf(layer.key);

    btn.innerHTML = `${layer.label}<span class="tab__count">${count}</span>`;
    btn.addEventListener('click', () => selectLayer(layer.key));
    host.appendChild(btn);
  });

  host.addEventListener('keydown', (e) => {
    const keys = ['ArrowRight', 'ArrowLeft', 'Home', 'End'];
    if (!keys.includes(e.key)) return;
    const tabs = [...host.querySelectorAll('.tab')];
    let i = tabs.findIndex((t) => t.dataset.key === state.layerKey);
    if (e.key === 'ArrowRight') i = (i + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') i = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') i = 0;
    else i = tabs.length - 1;
    e.preventDefault();
    selectLayer(tabs[i].dataset.key);
    tabs[i].focus();
  });
}

/** Channel count for a layer, read off the pre-computed plan. */
function channelsOf(key) {
  const L = state.model.plan.find((p) => p.name === key);
  return L ? L.shape[L.shape.length - 1] : 0;
}

function selectLayer(key) {
  state.layerKey = key;
  const tabs = [...$('feat-tabs').querySelectorAll('.tab')];
  tabs.forEach((t) => {
    const on = t.dataset.key === key;
    t.setAttribute('aria-selected', on ? 'true' : 'false');
    t.tabIndex = on ? 0 : -1;
    if (on) $('tiles').setAttribute('aria-labelledby', t.id);
  });
  renderFeatureMaps();
}

/** Grows or shrinks the pool of tile canvases to exactly `n`. */
function ensureTiles(n, w, h) {
  const host = $('tiles');
  while (tilePool.length < n) {
    const c = makeCanvas(w, h);
    tilePool.push({ canvas: c, ctx: c.getContext('2d') });
    host.appendChild(c);
  }
  for (let i = 0; i < tilePool.length; i++) {
    const t = tilePool[i];
    if (i < n) {
      if (t.canvas.width !== w || t.canvas.height !== h) {
        t.canvas.width = w;
        t.canvas.height = h;
      }
      if (!t.canvas.isConnected) host.appendChild(t.canvas);
    } else if (t.canvas.isConnected) {
      t.canvas.remove();
    }
  }
}

function renderFeatureMaps() {
  if (!state.activations) return;   // called once before the first inference
  const layer = FEATURE_LAYERS.find((l) => l.key === state.layerKey);
  $('feat-caption').textContent = layer.caption;
  const host = $('tiles');

  if (layer.key === '__input__') {
    host.classList.add('tiles--input');
    $('feat-meta').textContent = 'input · 32×32×3 · rgb, scaled to 0–1';
    ensureTiles(1, SIZE, SIZE);
    const t = tilePool[0];
    t.ctx.imageSmoothingEnabled = false;
    t.ctx.drawImage(scratchFromFloat(state.input), 0, 0);
    tilePool[0].canvas.title = 'input';
    return;
  }

  host.classList.remove('tiles--input');
  const act = state.activations.get(layer.key);
  const [h, w, c] = act.shape;
  const plan = state.model.plan.find((p) => p.name === layer.key);

  ensureTiles(c, w, h);

  const data = act.data;
  const plane = h * w;
  let silent = 0;

  for (let ch = 0; ch < c; ch++) {
    // Independent min-max per channel, otherwise deeper layers render as black.
    let min = Infinity, max = -Infinity;
    for (let p = 0, k = ch; p < plane; p++, k += c) {
      const v = data[k];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const span = max - min;
    const scale = span > 1e-9 ? 1 / span : 0;
    if (scale === 0) silent++;

    const t = tilePool[ch];
    const img = t.ctx.createImageData(w, h);
    const px = img.data;
    for (let p = 0, k = ch, o = 0; p < plane; p++, k += c, o += 4) {
      // Square root ramp: still 0 -> black and max -> white, but it lifts weak
      // responses into view. These maps are mostly exact zeros, so a linear ramp
      // leaves the deeper layers looking empty. Zeros stay zero either way.
      const g = 255 * Math.sqrt((data[k] - min) * scale);
      px[o] = g; px[o + 1] = g; px[o + 2] = g; px[o + 3] = 255;
    }
    t.ctx.putImageData(img, 0, 0);
    t.canvas.title = silentTitle(ch, min, max, scale === 0);
  }

  $('feat-meta').textContent =
    `${layer.key} · ${h}×${w}×${c} · ${plan.activation || 'linear'} · ` +
    `${silent}/${c} channels silent on this glyph · per-tile min-max, √ ramp`;
}

function silentTitle(ch, min, max, dead) {
  return dead
    ? `channel ${ch} · silent (all ${min.toFixed(2)})`
    : `channel ${ch} · range ${min.toFixed(2)} to ${max.toFixed(2)}`;
}

/* ------------------------------------------------------------------ *
 * 4. Occlusion sensitivity
 * ------------------------------------------------------------------ */

const heatSmall = makeCanvas(GRID, GRID);
const heatSmallCtx = heatSmall.getContext('2d');

function initOcclusion() {
  $('occl-btn').addEventListener('click', runOcclusion);
}

function cancelOcclusion() {
  state.occlToken++;
  state.occlRunning = false;
  $('occl-btn').disabled = false;
  $('occl-progress').hidden = true;
  $('occl-legend').hidden = true;
  $('occl-status').textContent = '';
  $('heat-canvas').classList.remove('on');
}

function runOcclusion() {
  if (state.occlRunning) return;
  const token = ++state.occlToken;
  state.occlRunning = true;

  const model = state.model;
  const base = Float32Array.from(state.probs);
  const target = argmax(base);
  const baseP = base[target];
  const className = model.classes[target];

  const src = state.input;
  const work = new Float32Array(src.length);
  const drops = new Float32Array(GRID * GRID);

  const btn = $('occl-btn');
  const status = $('occl-status');
  const prog = $('occl-progress');
  const bar = $('occl-progress-bar');

  btn.disabled = true;
  prog.hidden = false;
  bar.style.width = '0%';
  $('heat-canvas').classList.remove('on');
  status.textContent = `occluding “${className}” · 0/${GRID * GRID}`;

  let pos = 0;

  const step = () => {
    // A new glyph (or a new stroke) invalidates this run.
    if (token !== state.occlToken) return;

    const end = Math.min(pos + CHUNK, GRID * GRID);
    for (; pos < end; pos++) {
      work.set(src);
      const gy = (pos / GRID) | 0;
      const gx = pos % GRID;
      const y0 = gy * STRIDE;
      const x0 = gx * STRIDE;
      const y1 = Math.min(y0 + PATCH, SIZE);
      const x1 = Math.min(x0 + PATCH, SIZE);
      for (let y = y0; y < y1; y++) {
        let o = (y * SIZE + x0) * CH;
        for (let x = x0; x < x1; x++) {
          work[o] = OCCL_GREY;
          work[o + 1] = OCCL_GREY;
          work[o + 2] = OCCL_GREY;
          o += CH;
        }
      }
      // Pooled buffers: read the one number we need before the next call.
      drops[pos] = baseP - model.predict(work)[target];
    }

    bar.style.width = ((pos / (GRID * GRID)) * 100).toFixed(0) + '%';
    status.textContent = `occluding “${className}” · ${pos}/${GRID * GRID}`;

    if (pos < GRID * GRID) {
      requestAnimationFrame(step);
    } else {
      finishOcclusion(drops, className, baseP);
    }
  };

  requestAnimationFrame(step);
}

function finishOcclusion(drops, className, baseP) {
  let peak = 0;
  for (let i = 0; i < drops.length; i++) if (drops[i] > peak) peak = drops[i];
  const scale = peak > 1e-6 ? 1 / peak : 0;

  const img = heatSmallCtx.createImageData(GRID, GRID);
  const px = img.data;
  for (let i = 0, o = 0; i < drops.length; i++, o += 4) {
    const t = Math.max(0, drops[i]) * scale;
    const [r, g, b] = rampColor(t);
    px[o] = r; px[o + 1] = g; px[o + 2] = b;
    // Keep low-importance regions faint so the glyph stays readable underneath.
    px[o + 3] = (0.35 + 0.65 * t) * 255;
  }
  heatSmallCtx.putImageData(img, 0, 0);

  const heat = $('heat-canvas');
  const ctx = heat.getContext('2d');
  ctx.clearRect(0, 0, heat.width, heat.height);
  ctx.imageSmoothingEnabled = true;   // smooth, as specified
  ctx.imageSmoothingQuality = 'high';
  ctx.globalAlpha = 0.62;
  ctx.drawImage(heatSmall, 0, 0, heat.width, heat.height);
  ctx.globalAlpha = 1;
  heat.classList.add('on');

  $('occl-progress').hidden = true;
  $('occl-legend').hidden = false;
  $('occl-btn').disabled = false;
  state.occlRunning = false;
  $('occl-status').textContent =
    `64 passes · “${className}” started at ${(baseP * 100).toFixed(1)}% · ` +
    `worst patch cost ${(peak * 100).toFixed(1)} points`;
}

/* ------------------------------------------------------------------ *
 * 5. Draw mode
 * ------------------------------------------------------------------ */

const ds = makeCanvas(SIZE, SIZE);
const dsCtx = ds.getContext('2d', { willReadFrequently: true });

let drawCtx = null;
let drawing = false;
let dirty = false;
let hasStrokes = false;

function initDrawMode() {
  const pad = $('draw-canvas');
  drawCtx = pad.getContext('2d', { willReadFrequently: true });
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';
  drawCtx.lineWidth = 14;
  // Dataset ink is mid-grey rather than pure black; match it roughly.
  drawCtx.strokeStyle = '#3f3f3f';

  const pos = (e) => {
    const r = pad.getBoundingClientRect();
    return [
      (e.clientX - r.left) * (pad.width / r.width),
      (e.clientY - r.top) * (pad.height / r.height),
    ];
  };

  pad.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    try { pad.setPointerCapture(e.pointerId); } catch (_) { /* not fatal */ }
    drawing = true;
    hasStrokes = true;
    const [x, y] = pos(e);
    drawCtx.beginPath();
    drawCtx.moveTo(x, y);
    // A dot, so a single tap leaves a mark.
    drawCtx.lineTo(x + 0.01, y);
    drawCtx.stroke();
    schedule();
  });

  pad.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    const [x, y] = pos(e);
    drawCtx.lineTo(x, y);
    drawCtx.stroke();
    schedule();
  });

  const stop = () => {
    if (!drawing) return;
    drawing = false;
    schedule();
  };
  // Pointer capture keeps the stroke alive outside the pad, so no leave handler.
  pad.addEventListener('pointerup', stop);
  pad.addEventListener('pointercancel', stop);

  $('draw-clear').addEventListener('click', () => {
    drawCtx.clearRect(0, 0, pad.width, pad.height);
    hasStrokes = false;
    selectSample(state.sampleIndex);   // fall back to the gallery glyph
  });
}

/** Coalesce inference to one run per animation frame while drawing. */
function schedule() {
  if (dirty) return;
  dirty = true;
  requestAnimationFrame(() => {
    dirty = false;
    runOnDrawing();
  });
}

function runOnDrawing() {
  if (!hasStrokes) return;
  const pad = $('draw-canvas');

  // The pad is transparent where unpainted; composite it over white so the
  // downsampled glyph matches the dataset's ink-on-paper polarity.
  dsCtx.fillStyle = '#ffffff';
  dsCtx.fillRect(0, 0, SIZE, SIZE);
  dsCtx.imageSmoothingEnabled = true;
  dsCtx.imageSmoothingQuality = 'high';
  dsCtx.drawImage(pad, 0, 0, SIZE, SIZE);

  const d = dsCtx.getImageData(0, 0, SIZE, SIZE).data;
  const input = new Float32Array(SIZE * SIZE * CH);
  for (let p = 0, s = 0, t = 0; p < SIZE * SIZE; p++, s += 4, t += CH) {
    input[t] = d[s] / 255;
    input[t + 1] = d[s + 1] / 255;
    input[t + 2] = d[s + 2] / 255;
  }

  state.source = 'drawing';
  setInput(input);
}

/** Ghost the selected gallery glyph behind the drawing pad as a tracing guide. */
function drawGhost(i) {
  const ghost = $('ghost-canvas');
  const ctx = ghost.getContext('2d');
  ctx.clearRect(0, 0, ghost.width, ghost.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratchFromRGB(state.model.sampleRGB(i)), 0, 0, ghost.width, ghost.height);
}

/* ------------------------------------------------------------------ */

boot();
