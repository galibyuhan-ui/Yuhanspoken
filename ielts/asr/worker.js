// Yuhan speech-to-text worker: Whisper base (q8) running on-device with transformers.js.
// Large files are published in 5 MB parts and reassembled here; assembled files are kept in the Cache API when available.
import { pipeline, env } from './transformers.min.js';

const BASE = new URL('./', self.location.href).href;
const MODEL = 'Xenova/whisper-base';
const CACHE_NAME = 'yuhan-asr-v1';
let manifest = null, asr = null, loading = null;

const post = (m) => self.postMessage(m);

async function openCache() { try { return await caches.open(CACHE_NAME); } catch (e) { return null; } }

let doneBytes = 0, totalBytes = 0;
async function assemble(key, type) {
  const cacheKey = BASE + '__assembled__/' + key;
  const cache = await openCache();
  if (cache) { try { const hit = await cache.match(cacheKey); if (hit) { doneBytes += manifest[key].size; post({ type: 'progress', done: doneBytes, total: totalBytes }); return await hit.blob(); } } catch (e) {} }
  const parts = [];
  for (let i = 0; i < manifest[key].parts; i++) {
    const r = await fetch(BASE + key + '.' + i + '.wasm');
    if (!r.ok) throw new Error('Could not download part ' + key + '.' + i);
    const b = await r.arrayBuffer(); parts.push(b); doneBytes += b.byteLength;
    post({ type: 'progress', done: doneBytes, total: totalBytes });
  }
  const blob = new Blob(parts, { type });
  if (cache) { try { await cache.put(cacheKey, new Response(blob)); } catch (e) {} }
  return blob;
}

async function load() {
  if (asr) return asr;
  if (loading) return loading;
  loading = (async () => {
    manifest = await (await fetch(BASE + 'manifest.json')).json();
    totalBytes = Object.values(manifest).reduce((a, m) => a + m.size, 0); doneBytes = 0;
    post({ type: 'progress', done: 0, total: totalBytes });
    const wasmBlob = await assemble('wasm', 'application/wasm');
    const encBlob = await assemble('enc', 'application/octet-stream');
    const decBlob = await assemble('dec', 'application/octet-stream');
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.useBrowserCache = false;
    env.remoteHost = BASE + 'm/';
    env.remotePathTemplate = '';
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.proxy = false;
    env.backends.onnx.wasm.wasmPaths = { mjs: BASE + 'ort-wasm-simd-threaded.jsep.mjs', wasm: URL.createObjectURL(wasmBlob) };
    env.useCustomCache = true;
    env.customCache = {
      async match(req) {
        const url = typeof req === 'string' ? req : req.url;
        if (url.endsWith('encoder_model_quantized.onnx')) return new Response(encBlob);
        if (url.endsWith('decoder_model_merged_quantized.onnx')) return new Response(decBlob);
        return undefined;
      },
      async put() {}
    };
    post({ type: 'status', text: 'Starting the speech model…' });
    asr = await pipeline('automatic-speech-recognition', MODEL, { dtype: 'q8', device: 'wasm' });
    return asr;
  })();
  try { return await loading; } catch (e) { loading = null; throw e; }
}

function split(a, sr = 16000, max = 28) {
  const out = []; let s = 0;
  while (s < a.length) {
    let e = Math.min(a.length, s + max * sr);
    if (e < a.length) {
      let best = e, bv = Infinity; const win = Math.floor(0.02 * sr);
      for (let p = Math.max(s + sr, e - 8 * sr); p < e; p += win) { let v = 0; for (let i = p; i < p + win; i++) v += Math.abs(a[i]); if (v < bv) { bv = v; best = p + (win >> 1); } }
      e = best;
    }
    out.push(a.subarray(s, e)); s = e;
  }
  return out;
}

self.onmessage = async (ev) => {
  const m = ev.data || {};
  try {
    if (m.type === 'preload') { await load(); post({ type: 'ready' }); return; }
    if (m.type === 'transcribe') {
      await load(); post({ type: 'ready' });
      const chunks = split(m.audio);
      const texts = [];
      for (let i = 0; i < chunks.length; i++) {
        post({ type: 'status', text: `Transcribing ${i + 1} of ${chunks.length}…`, part: i, parts: chunks.length });
        const o = await asr(chunks[i], { language: 'english', task: 'transcribe' });
        const t = (o.text || '').trim(); if (t) texts.push(t);
        post({ type: 'partial', text: texts.join(' ') });
      }
      post({ type: 'result', text: texts.join(' '), id: m.id });
    }
  } catch (e) {
    post({ type: 'error', message: String(e && e.message || e), id: m.id });
  }
};
