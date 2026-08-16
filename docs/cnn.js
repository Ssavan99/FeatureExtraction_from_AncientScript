/*
 * cnn.js — a dependency-free forward pass for the MODI / Devanagari character CNN.
 *
 * Everything runs on flat Float32Arrays in HWC (height, width, channel) layout.
 * Convolution kernels come straight out of Keras in [kh, kw, inChannels, outChannels]
 * order, C-contiguous, which conveniently puts outChannels in the fastest-moving
 * position — so the innermost accumulation loop walks contiguous memory.
 *
 * No WebGL, no WASM, no dependencies. The network is ~33M multiply-accumulates,
 * which lands around 25-50 ms per forward pass in scalar JavaScript on a laptop.
 * Two things do most of the work to keep that down: batch-norm layers are folded
 * into the following convolution at load time (see _foldBatchNorm), and the conv
 * inner loop skips zero inputs, which is worth a lot because ReLU leaves 75-99%
 * of these activation maps at exactly zero.
 */

/* ------------------------------------------------------------------ *
 * Activations
 * ------------------------------------------------------------------ */

/** In-place ReLU. */
export function relu(a) {
  for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0;
  return a;
}

/**
 * In-place softmax. Subtracts the max before exponentiating so that large
 * logits cannot overflow to Infinity (which would produce NaN after the divide).
 */
export function softmax(a) {
  let max = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > max) max = a[i];
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.exp(a[i] - max);
    a[i] = e;
    sum += e;
  }
  const inv = 1 / sum;
  for (let i = 0; i < a.length; i++) a[i] *= inv;
  return a;
}

function applyActivation(buf, activation) {
  if (activation === 'relu') return relu(buf);
  if (activation === 'softmax') return softmax(buf);
  return buf; // "linear"
}

/* ------------------------------------------------------------------ *
 * Ops
 * ------------------------------------------------------------------ */

/**
 * 2-D convolution, valid padding, stride 1.
 *
 * input : Float32Array, shape [inH, inW, inC] (HWC)
 * kernel: Float32Array, shape [kh, kw, inC, outC]
 * out   : Float32Array, shape [inH-kh+1, inW-kw+1, outC]
 *
 * The accumulator is allocated by the caller and reused across output pixels so
 * that nothing is allocated inside the loop nest.
 */
export function conv2d(input, inH, inW, inC, kernel, bias, kh, kw, outC, activation, out, acc) {
  const outH = inH - kh + 1;
  const outW = inW - kw + 1;
  const rowStride = inW * inC;   // bytes-per-input-row, in floats
  const kStep = inC * outC;      // kernel stride per (ky, kx) tap
  const doRelu = activation === 'relu';

  let o = 0;
  for (let oy = 0; oy < outH; oy++) {
    const rowBase = oy * rowStride;
    for (let ox = 0; ox < outW; ox++) {
      acc.set(bias);
      let kBase = 0;
      const colBase = rowBase + ox * inC;
      for (let ky = 0; ky < kh; ky++) {
        let iBase = colBase + ky * rowStride;
        for (let kx = 0; kx < kw; kx++) {
          for (let ic = 0; ic < inC; ic++) {
            const v = input[iBase + ic];
            // ReLU makes activations sparse; skipping zeros is a real win here.
            if (v !== 0) {
              const kOff = kBase + ic * outC;
              for (let oc = 0; oc < outC; oc++) acc[oc] += v * kernel[kOff + oc];
            }
          }
          iBase += inC;
          kBase += kStep;
        }
      }
      if (doRelu) {
        for (let oc = 0; oc < outC; oc++) {
          const v = acc[oc];
          out[o + oc] = v > 0 ? v : 0;
        }
      } else {
        for (let oc = 0; oc < outC; oc++) out[o + oc] = acc[oc];
      }
      o += outC;
    }
  }
  return out;
}

/**
 * Per-channel batch normalisation, in place.
 *
 * The maths is gamma * (x - mean) / sqrt(variance + epsilon) + beta, but the
 * per-channel scale and shift are folded once at load time into
 *   scale = gamma / sqrt(variance + epsilon)
 *   shift = beta - mean * scale
 * so inference is a single multiply-add per element.
 */
export function batchNorm(x, channels, scale, shift) {
  const n = x.length;
  for (let i = 0; i < n; i += channels) {
    for (let c = 0; c < channels; c++) x[i + c] = x[i + c] * scale[c] + shift[c];
  }
  return x;
}

/** 2x2 max pool, stride 2, HWC. Odd trailing rows/cols are dropped (Keras behaviour). */
export function maxPool2d(input, inH, inW, C, out) {
  const outH = inH >> 1;
  const outW = inW >> 1;
  const rowStride = inW * C;
  let o = 0;
  for (let oy = 0; oy < outH; oy++) {
    const r0 = (oy * 2) * rowStride;
    const r1 = r0 + rowStride;
    for (let ox = 0; ox < outW; ox++) {
      const c0 = ox * 2 * C;
      const a = r0 + c0;
      const b = a + C;
      const c = r1 + c0;
      const d = c + C;
      for (let ch = 0; ch < C; ch++) {
        let m = input[a + ch];
        const v1 = input[b + ch];
        if (v1 > m) m = v1;
        const v2 = input[c + ch];
        if (v2 > m) m = v2;
        const v3 = input[d + ch];
        if (v3 > m) m = v3;
        out[o + ch] = m;
      }
      o += C;
    }
  }
  return out;
}

/** Global average pool over H and W -> length-C vector. */
export function globalAveragePool(input, inH, inW, C, out) {
  out.fill(0);
  const n = inH * inW;
  for (let i = 0; i < input.length; i += C) {
    for (let c = 0; c < C; c++) out[c] += input[i + c];
  }
  const inv = 1 / n;
  for (let c = 0; c < C; c++) out[c] *= inv;
  return out;
}

/** Fully connected layer. kernel is [inN, outN], C-contiguous. */
export function dense(input, inN, kernel, bias, outN, activation, out) {
  out.set(bias);
  for (let i = 0; i < inN; i++) {
    const v = input[i];
    if (v === 0) continue;
    const kOff = i * outN;
    for (let j = 0; j < outN; j++) out[j] += v * kernel[kOff + j];
  }
  return applyActivation(out, activation);
}

/* ------------------------------------------------------------------ *
 * Model
 * ------------------------------------------------------------------ */

function joinPath(base, file) {
  if (!base) return file;
  return base.endsWith('/') ? base + file : base + '/' + file;
}

async function fetchOk(url, kind) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (HTTP ${res.status})`);
  return kind === 'json' ? res.json() : res.arrayBuffer();
}

export class ModiCNN {
  /**
   * Loads the manifest, the float32 weight blob and the sample glyph bytes.
   * `base` must stay relative so the page works from a GitHub Pages subpath.
   */
  static async load(base = 'weights', onProgress) {
    const step = (msg) => { if (onProgress) onProgress(msg); };
    step('Reading model manifest');
    const manifest = await fetchOk(joinPath(base, 'model2.json'), 'json');
    step('Downloading weights');
    const [weights, samples] = await Promise.all([
      fetchOk(joinPath(base, 'model2.bin'), 'buffer'),
      fetchOk(joinPath(base, 'samples.bin'), 'buffer'),
    ]);
    step('Preparing network');
    return new ModiCNN(manifest, weights, samples);
  }

  constructor(manifest, weightsBuffer, samplesBuffer) {
    this.manifest = manifest;
    this.classes = manifest.classes;
    this.samples = manifest.samples;
    this.inputShape = manifest.inputShape;
    this.sampleShape = manifest.sampleShape || manifest.inputShape;

    // Zero-copy views into the weight blob. Every offset is 4-byte aligned.
    this.tensors = manifest.tensors.map(
      (t) => new Float32Array(weightsBuffer, t.offset, t.size)
    );

    const [sh, sw, sc] = this.sampleShape;
    this.sampleBytes = new Uint8Array(samplesBuffer);
    this.sampleStride = sh * sw * sc;
    this.sampleCount = Math.floor(this.sampleBytes.length / this.sampleStride);

    this.plan = this._buildPlan();
    this._foldBatchNorm();
    this._pool = this.plan.map((l) => new Float32Array(l.outSize));
    // Largest channel count across conv layers, for the shared accumulator.
    const maxAcc = this.plan.reduce((m, l) => Math.max(m, l.acc || 0), 1);
    this._acc = new Float32Array(maxAcc);
  }

  /**
   * Walks the layer list once, resolving tensor indices to typed-array views,
   * folding batch-norm parameters, and computing every intermediate shape so
   * output buffers can be sized ahead of time.
   */
  _buildPlan() {
    let [h, w, c] = this.inputShape;
    return this.manifest.layers.map((layer) => {
      const entry = { name: layer.name, type: layer.type, activation: layer.activation };
      switch (layer.type) {
        case 'Conv2D': {
          const [kh, kw, inC, outC] = layer.kernelShape;
          entry.kernel = this.tensors[layer.kernel];
          entry.bias = this.tensors[layer.bias];
          entry.kh = kh; entry.kw = kw; entry.inC = inC; entry.outC = outC;
          entry.inH = h; entry.inW = w;
          h = h - kh + 1; w = w - kw + 1; c = outC;
          entry.acc = outC;
          break;
        }
        case 'BatchNormalization': {
          const gamma = this.tensors[layer.gamma];
          const beta = this.tensors[layer.beta];
          const mean = this.tensors[layer.mean];
          const variance = this.tensors[layer.variance];
          const eps = layer.epsilon;
          const n = layer.channels;
          const scale = new Float32Array(n);
          const shift = new Float32Array(n);
          for (let i = 0; i < n; i++) {
            scale[i] = gamma[i] / Math.sqrt(variance[i] + eps);
            shift[i] = beta[i] - mean[i] * scale[i];
          }
          entry.scale = scale; entry.shift = shift; entry.channels = n;
          break;
        }
        case 'MaxPooling2D': {
          entry.inH = h; entry.inW = w; entry.C = c;
          h = h >> 1; w = w >> 1;
          break;
        }
        case 'GlobalAveragePooling2D': {
          entry.inH = h; entry.inW = w; entry.C = c;
          h = 1; w = 1;
          break;
        }
        case 'Dense': {
          const [inN, outN] = layer.kernelShape;
          entry.kernel = this.tensors[layer.kernel];
          entry.bias = this.tensors[layer.bias];
          entry.inN = inN; entry.outN = outN;
          c = outN;
          break;
        }
        default:
          throw new Error(`Unsupported layer type: ${layer.type}`);
      }
      // Spatial layers keep [h, w, c]; pooled/dense layers collapse to [c].
      entry.shape = (layer.type === 'GlobalAveragePooling2D' || layer.type === 'Dense')
        ? [c]
        : [h, w, c];
      entry.outSize = entry.shape.reduce((a, b) => a * b, 1);
      return entry;
    });
  }

  /**
   * Folds every BatchNormalization layer into the Conv2D that follows it.
   *
   * Batch norm is the affine map  y[c] = scale[c] * x[c] + shift[c],  so for a
   * convolution over y:
   *
   *   sum_p,c  k[p,c,oc] * (scale[c]*x[p,c] + shift[c])
   *     = sum_p,c (k[p,c,oc]*scale[c]) * x[p,c]  +  sum_p,c k[p,c,oc]*shift[c]
   *
   * The second sum has no dependence on position, so it collapses into the
   * convolution's bias. (This holds because padding is 'valid' — every output
   * position sees a complete patch. With zero padding the border terms would
   * differ and the fold would be wrong.)
   *
   * This is algebraically exact, and the real prize is not the removed batch-norm
   * pass: it is that the convolution now reads the *ReLU* output directly, which
   * is 75-99% zeros here. Batch norm shifts those zeros to arbitrary values and
   * destroys the sparsity the conv inner loop exploits. Folding is worth ~4x on
   * the affected layers.
   */
  _foldBatchNorm() {
    for (let i = 0; i < this.plan.length - 1; i++) {
      const bn = this.plan[i];
      const cv = this.plan[i + 1];
      if (bn.type !== 'BatchNormalization' || cv.type !== 'Conv2D') continue;
      if (bn.channels !== cv.inC) continue;

      const { kh, kw, inC, outC } = cv;
      const k = cv.kernel;
      const kf = new Float32Array(k.length);
      const bf = Float32Array.from(cv.bias);

      for (let p = 0; p < kh * kw; p++) {
        for (let ic = 0; ic < inC; ic++) {
          const base = (p * inC + ic) * outC;
          const s = bn.scale[ic];
          const sh = bn.shift[ic];
          for (let oc = 0; oc < outC; oc++) {
            const w = k[base + oc];
            kf[base + oc] = w * s;
            bf[oc] += w * sh;
          }
        }
      }

      cv.kernel = kf;
      cv.bias = bf;
      bn.folded = true;   // still computed when collecting, for the explorer
    }
  }

  /** Raw uint8 RGB bytes for sample `i` (length 32*32*3). */
  sampleRGB(i) {
    const off = i * this.sampleStride;
    return this.sampleBytes.subarray(off, off + this.sampleStride);
  }

  /** Sample `i` scaled to [0,1] as model input. */
  sampleInput(i) {
    const src = this.sampleRGB(i);
    const out = new Float32Array(src.length);
    for (let k = 0; k < src.length; k++) out[k] = src[k] / 255;
    return out;
  }

  /**
   * Runs the network.
   *
   * `collect: true`  -> every layer gets a freshly allocated output buffer and
   *                     they are all returned, keyed by layer name.
   * `collect: false` -> pooled buffers are reused. Much less GC churn, which
   *                     matters for the 64-pass occlusion sweep, but the caller
   *                     must read what it needs before calling forward() again.
   */
  forward(input, { collect = true } = {}) {
    let x = input;
    let h = this.inputShape[0], w = this.inputShape[1];
    const activations = collect ? new Map() : null;

    for (let i = 0; i < this.plan.length; i++) {
      const L = this.plan[i];

      // A folded batch norm is already baked into the next conv's weights, so it
      // is skipped in the compute path. When collecting we still materialise its
      // output for the feature-map explorer, but `x` deliberately stays pointing
      // at the pre-normalisation (sparse) tensor that the next conv expects.
      if (L.folded) {
        if (collect) {
          const bnOut = new Float32Array(L.outSize);
          bnOut.set(x);
          batchNorm(bnOut, L.channels, L.scale, L.shift);
          activations.set(L.name, { name: L.name, type: L.type, shape: L.shape, data: bnOut });
        }
        continue;
      }

      let out = collect ? new Float32Array(L.outSize) : this._pool[i];

      switch (L.type) {
        case 'Conv2D':
          conv2d(x, L.inH, L.inW, L.inC, L.kernel, L.bias, L.kh, L.kw, L.outC,
            L.activation, out, this._acc);
          h = L.shape[0]; w = L.shape[1];
          break;
        case 'BatchNormalization':
          // Batch norm is elementwise: copy then scale, so the previous layer's
          // buffer stays intact for the feature-map explorer.
          out.set(x);
          batchNorm(out, L.channels, L.scale, L.shift);
          break;
        case 'MaxPooling2D':
          maxPool2d(x, L.inH, L.inW, L.C, out);
          h = L.shape[0]; w = L.shape[1];
          break;
        case 'GlobalAveragePooling2D':
          globalAveragePool(x, L.inH, L.inW, L.C, out);
          h = 1; w = 1;
          break;
        case 'Dense':
          dense(x, L.inN, L.kernel, L.bias, L.outN, L.activation, out);
          break;
      }

      if (collect) {
        activations.set(L.name, { name: L.name, type: L.type, shape: L.shape, data: out });
      }
      x = out;
    }

    return { probs: x, activations };
  }

  /** Convenience: probability of every class, without keeping intermediates. */
  predict(input) {
    return this.forward(input, { collect: false }).probs;
  }

  /** Indices of the k highest probabilities, descending. */
  topK(probs, k = 5) {
    const idx = Array.from(probs.keys());
    idx.sort((a, b) => probs[b] - probs[a]);
    return idx.slice(0, k);
  }
}

export default ModiCNN;
