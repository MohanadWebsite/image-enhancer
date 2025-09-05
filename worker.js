// worker.js — updated: local ort + wasmPaths + robust backend fallback + ImageBitmap loading
// Make sure you uploaded /onnx/ort.min.js and the wasm/mjs files into /onnx/ in your repo.

try {
  importScripts('/onnx/ort.min.js');
} catch (e) {
  // fallback to CDN if local import fails (useful for debugging)
  try {
    importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/ort.min.js');
    // note: if you use CDN, ensure its version matches the wasm files you uploaded.
  } catch (ee) {
    // If import fails completely, we must notify main thread
    self.postMessage({ type: 'error', msg: 'Failed to import ort.min.js: ' + ee.message });
  }
}

let sessions = { esrgan: null, face: null };
let cfg = { scale: 4, tileSize: 512, overlap: 16, inputName: 'input', outputName: null, normalize: { method: '0-1' }, format: 'image/jpeg', quality: 0.92 };

//
// Configure wasm paths and threads if possible
//
if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
  try {
    ort.env.wasm.wasmPaths = '/onnx/'; // where you uploaded the .mjs/.wasm files
    // set a reasonable default for number of threads (if SharedArrayBuffer available)
    try {
      const hc = typeof navigator !== 'undefined' && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 2;
      // use up to (hc - 1) threads but clamp it
      ort.env.wasm.numThreads = Math.max(1, Math.min(4, Math.floor(Math.max(1, hc) - 1)));
    } catch (_) { /* ignore */ }
    postMessage({ type: 'log', msg: 'ort.env.wasm configured, wasmPaths=/onnx/' });
  } catch (e) {
    postMessage({ type: 'log', msg: 'Could not configure ort.env.wasm: ' + e.message });
  }
} else {
  postMessage({ type: 'log', msg: 'ort global not found or ort.env.wasm missing (import failed?)' });
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      cfg = { ...cfg, ...(msg.options || {}) };
      postMessage({ type: 'log', msg: 'worker init: options merged' });

      if (msg.modelPaths && msg.modelPaths.realesrgan) {
        postMessage({ type: 'progress', index: -1, progress: 2, status: 'loading ESReGAN model...' });
        sessions.esrgan = await createOrtSession(msg.modelPaths.realesrgan);
        postMessage({ type: 'log', msg: 'ESRGAN session ready' });
      } else {
        postMessage({ type: 'log', msg: 'No realesrgan path provided in init' });
      }

      if (msg.modelPaths && msg.modelPaths.face_restore) {
        try {
          postMessage({ type: 'progress', index: -1, progress: 5, status: 'loading face restore model...' });
          sessions.face = await createOrtSession(msg.modelPaths.face_restore);
          postMessage({ type: 'log', msg: 'Face session ready' });
        } catch (e) {
          postMessage({ type: 'log', msg: 'Face model load failed: ' + e.message });
        }
      }

      postMessage({ type: 'ready' });
    } else if (msg.type === 'process') {
      const idx = msg.index;
      postMessage({ type: 'progress', index: idx, progress: 2, status: 'worker: loading image' });
      const img = await loadImageFromDataURL(msg.dataURL);

      const tiles = createTiles(img, cfg.tileSize, cfg.overlap);
      const outTiles = [];

      for (let t = 0; t < tiles.length; t++) {
        const tile = tiles[t];
        postMessage({ type: 'progress', index: idx, progress: 5 + Math.round((t / tiles.length) * 70), status: `processing tile ${t + 1}/${tiles.length}` });

        if (!sessions.esrgan) {
          outTiles.push({ x: tile.x, y: tile.y, canvas: tile.canvas });
          continue;
        }

        const inputTensor = canvasToOrtTensor(tile.canvas, cfg);
        const feeds = {}; feeds[cfg.inputName || 'input'] = inputTensor;
        const results = await sessions.esrgan.run(feeds);
        const outKey = cfg.outputName || Object.keys(results)[0];
        const outTensor = results[outKey];
        const outW = tile.w * cfg.scale; const outH = tile.h * cfg.scale;
        const outCanvas = tensorToOffscreenCanvas(outTensor, outW, outH);
        outTiles.push({ x: tile.x, y: tile.y, canvas: outCanvas });
      }

      postMessage({ type: 'progress', index: idx, progress: 85, status: 'merging tiles' });
      const merged = mergeTilesWeighted(outTiles, img.width * cfg.scale, img.height * cfg.scale, cfg.overlap * cfg.scale);

      postMessage({ type: 'progress', index: idx, progress: 90, status: 'post-processing' });

      let finalCanvas = merged;
      if (sessions.face) {
        try {
          const faceTensor = canvasToOrtTensor(merged, cfg);
          const faceRes = await sessions.face.run({ [cfg.inputName || 'input']: faceTensor });
          const outKey = cfg.outputName || Object.keys(faceRes)[0];
          finalCanvas = tensorToOffscreenCanvas(faceRes[outKey], merged.width, merged.height);
        } catch (e) { postMessage({ type: 'log', msg: 'Face restore failed: ' + e.message }); }
      }

      try { applyUnsharpMask(finalCanvas, 0.5, 1); } catch (e) { /* ignore */ }

      // convert to blob and return dataURL
      const blob = await (typeof finalCanvas.convertToBlob === 'function' ? finalCanvas.convertToBlob({ type: cfg.format, quality: cfg.quality }) : canvasToBlobFallback(finalCanvas, cfg.format, cfg.quality));
      const dataURL = await blobToDataURL(blob);
      postMessage({ type: 'result', index: idx, dataURL });
    }
  } catch (err) {
    postMessage({ type: 'error', msg: err && err.message ? err.message : String(err) });
  }
};

//
// Create ONNX session with fallback: WebGPU -> WASM
//
async function createOrtSession(path) {
  if (typeof ort === 'undefined') throw new Error('ONNX Runtime (ort) is not loaded in worker.');

  // Try WebGPU first (fast) then WASM fallback
  try {
    postMessage({ type: 'log', msg: 'Trying to create session with WebGPU backend...' });
    const sess = await ort.InferenceSession.create(path, { executionProviders: ['webgpu'] });
    postMessage({ type: 'log', msg: 'Session created using WebGPU' });
    return sess;
  } catch (eWebgpu) {
    postMessage({ type: 'log', msg: 'WebGPU session failed: ' + (eWebgpu && eWebgpu.message ? eWebgpu.message : eWebgpu) });
    // Try wasm (library will pick simd-threaded if available & allowed)
    try {
      postMessage({ type: 'log', msg: 'Trying to create session with WASM backend...' });
      const sess2 = await ort.InferenceSession.create(path, { executionProviders: ['wasm'] });
      postMessage({ type: 'log', msg: 'Session created using WASM backend' });
      return sess2;
    } catch (eWasm) {
      postMessage({ type: 'log', msg: 'WASM session creation failed: ' + (eWasm && eWasm.message ? eWasm.message : eWasm) });
      throw new Error('Failed to create ONNX session: ' + (eWasm && eWasm.message ? eWasm.message : eWasm));
    }
  }
}

//
// Image loading inside worker using createImageBitmap
//
function loadImageFromDataURL(dataURL) {
  return new Promise(async (resolve, reject) => {
    try {
      const res = await fetch(dataURL);
      const blob = await res.blob();
      const bmp = await createImageBitmap(blob);
      // createImageBitmap returns ImageBitmap which works with drawImage on OffscreenCanvas
      resolve(bmp);
    } catch (e) {
      reject(e);
    }
  });
}

//
// Helpers: tiling, tensor conversion, merge, post-processing
//
function createTiles(img, tileSize, overlap) {
  const pw = img.width, ph = img.height; const tiles = [];
  const tmp = new OffscreenCanvas(pw, ph); const tctx = tmp.getContext('2d');
  tctx.drawImage(img, 0, 0);
  for (let y = 0; y < ph; y += tileSize - overlap) {
    for (let x = 0; x < pw; x += tileSize - overlap) {
      const w = Math.min(tileSize, pw - x); const h = Math.min(tileSize, ph - y);
      const c = new OffscreenCanvas(w, h); const ctx = c.getContext('2d'); ctx.drawImage(tmp, x, y, w, h, 0, 0, w, h);
      tiles.push({ x, y, w, h, canvas: c });
    }
  }
  return tiles;
}

function canvasToOrtTensor(canvas, cfgLocal) {
  const ctx = canvas.getContext('2d'); const w = canvas.width, h = canvas.height;
  const id = ctx.getImageData(0, 0, w, h); const d = id.data;
  const floats = new Float32Array(3 * w * h);
  let p = 0;
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        let v = d[i + (c === 0 ? 0 : (c === 1 ? 1 : 2))];
        if (cfgLocal.normalize && cfgLocal.normalize.method === '0-1') v = v / 255.0;
        else if (cfgLocal.normalize && cfgLocal.normalize.method === '-1-1') v = (v / 255.0) * 2.0 - 1.0;
        floats[p++] = v;
      }
    }
  }
  return new ort.Tensor('float32', floats, [1, 3, h, w]);
}

function tensorToOffscreenCanvas(tensor, outW, outH) {
  const arr = tensor.data;
  const c = new OffscreenCanvas(outW, outH); const ctx = c.getContext('2d');
  const id = ctx.createImageData(outW, outH); const out = id.data;
  const plane = outW * outH; let rIdx = 0, gIdx = plane, bIdx = plane * 2;
  for (let i = 0; i < plane; i++) {
    const r = Math.round(clamp(arr[rIdx++]) * 255);
    const g = Math.round(clamp(arr[gIdx++]) * 255);
    const b = Math.round(clamp(arr[bIdx++]) * 255);
    const px = i * 4; out[px] = r; out[px + 1] = g; out[px + 2] = b; out[px + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  return c;
}

function clamp(v) { if (v === undefined || Number.isNaN(v)) return 0; return Math.min(1, Math.max(0, v)); }

function mergeTilesWeighted(tiles, outW, outH, overlap) {
  const dest = new OffscreenCanvas(outW, outH); const dctx = dest.getContext('2d');
  const outId = dctx.createImageData(outW, outH); const outData = outId.data;
  const accR = new Float32Array(outW * outH); const accG = new Float32Array(outW * outH); const accB = new Float32Array(outW * outH);
  const accW = new Float32Array(outW * outH);

  for (const t of tiles) {
    const tw = t.canvas.width, th = t.canvas.height; const tctx = t.canvas.getContext('2d');
    const tId = tctx.getImageData(0, 0, tw, th); const td = tId.data;
    for (let yy = 0; yy < th; yy++) {
      for (let xx = 0; xx < tw; xx++) {
        const srcIdx = (yy * tw + xx) * 4;
        const gx = t.x * cfg.scale + xx; const gy = t.y * cfg.scale + yy;
        const dstIdx = gy * outW + gx;
        let wx = 1.0, wy = 1.0;
        if (overlap > 0) {
          const edgeFactorX = overlap ? (1 - (Math.min(xx, tw - 1 - xx) / Math.max(1, (overlap)))) : 0;
          const edgeFactorY = overlap ? (1 - (Math.min(yy, th - 1 - yy) / Math.max(1, (overlap)))) : 0;
          wx = 1.0 - edgeFactorX * 0.5;
          wy = 1.0 - edgeFactorY * 0.5;
        }
        const w = wx * wy;
        accR[dstIdx] += td[srcIdx] * w;
        accG[dstIdx] += td[srcIdx + 1] * w;
        accB[dstIdx] += td[srcIdx + 2] * w;
        accW[dstIdx] += w;
      }
    }
  }

  for (let i = 0; i < outW * outH; i++) {
    const w = accW[i] || 1;
    const r = Math.round(accR[i] / w);
    const g = Math.round(accG[i] / w);
    const b = Math.round(accB[i] / w);
    const px = i * 4; outData[px] = r; outData[px + 1] = g; outData[px + 2] = b; outData[px + 3] = 255;
  }
  dctx.putImageData(outId, 0, 0);
  return dest;
}

function applyUnsharpMask(canvas, amount = 0.5, radius = 1) {
  const ctx = canvas.getContext('2d');
  try {
    ctx.filter = `blur(${radius}px)`;
    const copy = new OffscreenCanvas(canvas.width, canvas.height);
    const cctx = copy.getContext('2d');
    cctx.filter = `blur(${radius}px)`;
    cctx.drawImage(canvas, 0, 0);
    ctx.filter = 'none';
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = amount;
    ctx.drawImage(copy, 0, 0);
    ctx.globalAlpha = 1.0; ctx.globalCompositeOperation = 'source-over';
  } catch (e) { /* ignore */ }
}

function blobToDataURL(blob) { return new Promise(res => { const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(blob); }); }

//
// Fallback blob conversion for OffscreenCanvas in older browsers
//
function canvasToBlobFallback(canvas, type = 'image/jpeg', quality = 0.92) {
  return new Promise((resolve, reject) => {
    try {
      // transfer to a bitmap in main thread isn't possible here; try to use convertToBlob if exists
      if (typeof canvas.convertToBlob === 'function') {
        canvas.convertToBlob({ type, quality }).then(resolve).catch(reject);
      } else {
        // as last resort, draw to an ImageBitmap and use a canvas from it
        try {
          const bitmapPromise = createImageBitmap(canvas);
          bitmapPromise.then(bitmap => {
            const tmp = new OffscreenCanvas(bitmap.width, bitmap.height);
            const ctx = tmp.getContext('2d');
            ctx.drawImage(bitmap, 0, 0);
            tmp.convertToBlob({ type, quality }).then(resolve).catch(reject);
          }).catch(reject);
        } catch (e) {
          reject(e);
        }
      }
    } catch (e) {
      reject(e);
    }
  });
}
