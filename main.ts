// main.ts — WebGPU only (no WASM), YOLOv8 persons, optimisé allocations + boucle vidéo

import type * as ortTypes from 'onnxruntime-web';

// ---------- Hyperparams ----------
const INPUT_SIZE = 640;
const NMS_IOU = 0.10;
const MAX_DETECTIONS = 100;
const TOPK_PER_HEAD = 400;
const SCORE_THRESH = 0.30;
const OBJ_THRESH   = 0.25;
const PERSON_THRESH= 0.25;
const MIN_SIDE_FRAC= 0.015;

let ort: typeof import('onnxruntime-web');
let session: ortTypes.InferenceSession | null = null;

let INPUT_LAYOUT: 'NCHW' | 'NHWC' = 'NHWC';
let INPUT_NAME = '';
const MIRROR = true;
const MODEL_PATH = '/yolov8n.onnx';

// ---------- DOM ----------
const videoEl = document.getElementById('cam') as HTMLVideoElement | null;
const overlay = document.getElementById('overlay') as HTMLCanvasElement | null;
const channelEl = document.getElementById('channel') as HTMLDivElement | null;

if (!videoEl) throw new Error('#cam introuvable (video)');
if (!overlay) throw new Error('#overlay introuvable (canvas)');

const overlayCtx = overlay.getContext('2d');
if (!overlayCtx) throw new Error('Context 2D indisponible pour #overlay');

// ---------- Canvases & buffers réutilisés ----------
const preprocessCanvas = document.createElement('canvas');
preprocessCanvas.width = INPUT_SIZE;
preprocessCanvas.height = INPUT_SIZE;
// Hint de perf : on lit des pixels à chaque frame
const pctx = preprocessCanvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D;

// Buffers réutilisés pour le tenseur
const SIZE = INPUT_SIZE * INPUT_SIZE;
const nhwcBuf = new Float32Array(3 * SIZE);
const nchwBuf = new Float32Array(3 * SIZE);

// State overlay/viewport
let lastDpr = 0;
let lastRectW = 0;
let lastRectH = 0;

// DPI-safe resize du canvas mais seulement quand nécessaire
function resizeOverlayToVideoIfNeeded() {
  const rect = overlay!.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  if (overlay!.width !== Math.floor(rect.width * dpr) ||
      overlay!.height !== Math.floor(rect.height * dpr) ||
      lastDpr !== dpr || lastRectW !== rect.width || lastRectH !== rect.height) {

    overlay!.style.width = '100%';
    overlay!.style.height = '100%';
    overlay!.width = Math.max(1, Math.floor(rect.width * dpr));
    overlay!.height = Math.max(1, Math.floor(rect.height * dpr));

    overlayCtx!.setTransform(dpr, 0, 0, dpr, 0, 0);

    lastDpr = dpr;
    lastRectW = rect.width;
    lastRectH = rect.height;
  }
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

// BUGFIX: areaB utilisait (ay2 - by1) -> doit être (by2 - by1)
function iou(a: number[], b: number[]): number {
  const [ax1, ay1, ax2, ay2] = a, [bx1, by1, bx2, by2] = b;
  const iw = ax2 < bx1 || bx2 < ax1 ? 0 : Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const ih = ay2 < by1 || by2 < ay1 ? 0 : Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = (ax2 - ax1) * (ay2 - ay1);
  const areaB = (bx2 - bx1) * (by2 - by1);
  const denom = areaA + areaB - inter;
  return denom <= 0 ? 0 : inter / (denom + 1e-6);
}

function nms(boxes: number[][], scores: number[], iouTh = NMS_IOU, limit = MAX_DETECTIONS): number[] {
  const order = scores.map((s, i) => [s, i] as const).sort((a, b) => b[0] - a[0]).map(([, i]) => i);
  const keep: number[] = [];
  for (let k = 0; k < order.length && keep.length < limit; k++) {
    const i = order[k];
    let ok = true;
    for (let j = 0; j < keep.length; j++) {
      if (iou(boxes[i], boxes[keep[j]]) > iouTh) { ok = false; break; }
    }
    if (ok) keep.push(i);
  }
  return keep;
}

// --- Accessor générique pour têtes 2D/3D/4D ---
type Accessor = { N: number; C: number; get: (c: number, i: number) => number; };

function makeAccessor(t: ortTypes.Tensor): Accessor | null {
  const dims = t.dims;
  const data = t.data as Float32Array;

  if (dims.length === 4) {
    const [d0,d1,d2,d3] = dims;
    if (d0 === 1 && d3 >= 5) {
      const H = d1, W = d2, C = d3, N = H * W;
      return { N, C, get: (ch,i) => { const x=i%W; const y=(i-x)/W; return data[(y*W+x)*C+ch]; } };
    }
    if (d0 === 1 && d1 >= 5) {
      const C = d1, H = d2, W = d3, N = H * W;
      return { N, C, get: (ch,i) => { const x=i%W; const y=(i-x)/W; return data[ch*H*W + y*W + x]; } };
    }
  } else if (dims.length === 3) {
    const [a,b,c] = dims;
    if (a === 1 && b >= 5) { const C=b, N=c; return { N, C, get: (ch,i)=>data[ch*N+i] }; }
    if (a === 1 && c >= 5) { const N=b, C=c; return { N, C, get: (ch,i)=>data[i*C+ch] }; }
    if (b === 1 && a >= 5) { const C=a, N=c; return { N, C, get: (ch,i)=>data[ch*N+i] }; }
    if (b === 1 && c >= 5) { const N=a, C=c; return { N, C, get: (ch,i)=>data[i*C+ch] }; }
  } else if (dims.length === 2) {
    const [a,b] = dims;
    if (a >= 5) { const C=a, N=b; return { N, C, get: (ch,i)=>data[ch*N+i] }; }
    if (b >= 5) { const N=a, C=b; return { N, C, get: (ch,i)=>data[i*C+ch] }; }
  }

  console.warn('makeAccessor: forme inattendue ignorée:', dims);
  return null;
}

// ---------- Décodage YOLOv8 ----------
const HEAD_GS = [INPUT_SIZE / 8, INPUT_SIZE / 16, INPUT_SIZE / 32].map(v => Math.round(v));
const HEAD_COUNTS = HEAD_GS.map(g => g * g);
const HEAD_OFFSETS = [0, HEAD_COUNTS[0], HEAD_COUNTS[0] + HEAD_COUNTS[1]];
const SUM_HEADS = HEAD_COUNTS.reduce((a,b)=>a+b,0);

function gridInfoForIndex(i: number) {
  if (i < HEAD_OFFSETS[1]) {
    const g = HEAD_GS[0]; const ii = i;
    const cx = ii % g, cy = (ii - cx) / g; return { g, cx, cy, stride: INPUT_SIZE / g };
  } else if (i < HEAD_OFFSETS[2]) {
    const g = HEAD_GS[1]; const ii = i - HEAD_OFFSETS[1];
    const cx = ii % g, cy = (ii - cx) / g; return { g, cx, cy, stride: INPUT_SIZE / g };
  } else {
    const g = HEAD_GS[2]; const ii = i - HEAD_OFFSETS[2];
    const cx = ii % g, cy = (ii - cx) / g; return { g, cx, cy, stride: INPUT_SIZE / g };
  }
}

type Letterbox = { sx:number, sy:number, dw:number, dh:number, rw:number, rh:number };

function decodeYoloV8(
  results: Record<string, ortTypes.Tensor>,
  letter: Letterbox
) {
  const outs = Object.values(results);
  const allBoxes:number[][] = [];
  const allScores:number[] = [];

  const minSidePx = Math.max(8, Math.min(letter.rw, letter.rh) * MIN_SIDE_FRAC);

  for (let tIdx = 0; tIdx < outs.length; tIdx++) {
    const acc = makeAccessor(outs[tIdx]);
    if (!acc) continue;
    const { N, C, get } = acc;

    const numClasses = C - 5;
    if (numClasses < 1) continue;

    // Stock temporaire par tête pour limite TOPK
    const headBoxes:number[][] = [];
    const headScores:number[] = [];

    for (let i=0; i<N; i++) {
      let g:number, cx:number, cy:number, stride:number;
      if (N === SUM_HEADS) {
        const gi = gridInfoForIndex(i);
        g = gi.g; cx = gi.cx; cy = gi.cy; stride = gi.stride;
      } else {
        g = (Math.sqrt(N) + 0.5) | 0; // floor
        if (g < 1) g = 1;
        stride = INPUT_SIZE / g;
        cx = i % g;
        cy = (i - cx) / g;
      }

      const tx = get(0,i),  ty = get(1,i),  tw = get(2,i),  th = get(3,i);
      const to = get(4,i);
      const tPerson = get(5, i); // classe 0 = person

      const pObj = sigmoid(to);
      if (pObj < OBJ_THRESH) continue;
      const pCls = sigmoid(tPerson);
      if (pCls < PERSON_THRESH) continue;

      const x = ((sigmoid(tx) * 2 - 0.5) + cx) * stride;
      const y = ((sigmoid(ty) * 2 - 0.5) + cy) * stride;
      const w =  ((sigmoid(tw) * 2) ** 2) * stride;
      const h =  ((sigmoid(th) * 2) ** 2) * stride;

      if (w < minSidePx || h < minSidePx) continue;

      const conf = pObj * pCls;
      if (conf < SCORE_THRESH) continue;

      const x1l = x - w*0.5, y1l = y - h*0.5, x2l = x + w*0.5, y2l = y + h*0.5;

      const invSx = 1 / letter.sx, invSy = 1 / letter.sy;
      let x1 = (x1l - letter.dw) * invSx;
      let y1 = (y1l - letter.dh) * invSy;
      let x2 = (x2l - letter.dw) * invSx;
      let y2 = (y2l - letter.dh) * invSy;

      if (x1 < 0) x1 = 0;
      if (y1 < 0) y1 = 0;
      if (x2 > letter.rw) x2 = letter.rw;
      if (y2 > letter.rh) y2 = letter.rh;

      if (!(x2 > x1 && y2 > y1)) continue;

      headBoxes.push([x1,y1,x2,y2]);
      headScores.push(conf);
    }

    if (headBoxes.length > TOPK_PER_HEAD) {
      const order = headScores.map((s,i)=>[s,i] as const)
        .sort((a,b)=>b[0]-a[0])
        .slice(0, TOPK_PER_HEAD)
        .map(([,i])=>i);
      for (let k=0; k<order.length; k++) {
        const idx = order[k];
        allBoxes.push(headBoxes[idx]);
        allScores.push(headScores[idx]);
      }
    } else {
      for (let k=0; k<headBoxes.length; k++) {
        allBoxes.push(headBoxes[k]);
        allScores.push(headScores[k]);
      }
    }
  }

  const keep = nms(allBoxes, allScores, NMS_IOU, MAX_DETECTIONS);
  const out = new Array(keep.length);
  for (let i=0; i<keep.length; i++) {
    const k = keep[i];
    out[i] = { box: allBoxes[k], score: allScores[k], cls: 0, label: 'person' as const };
  }
  return out;
}

// ---------- Dessin ----------
function drawDetections(dets: { box:number[], score:number, cls:number, label:string }[]) {
  const vw = Math.max(1, videoEl!.videoWidth);
  const vh = Math.max(1, videoEl!.videoHeight);

  const rect = overlay!.getBoundingClientRect();
  const cw = rect.width;
  const ch = rect.height;

  const scale = Math.max(cw / vw, ch / vh);
  const rw = vw * scale;
  const rh = vh * scale;
  const ox = (cw - rw) * 0.5;
  const oy = (ch - rh) * 0.5;

  overlayCtx!.clearRect(0, 0, cw, ch);
  overlayCtx!.lineWidth = 2;
  overlayCtx!.font = '16px monospace';
  overlayCtx!.textBaseline = 'top';

  let people = 0;

  for (let i=0; i<dets.length; i++) {
    const det = dets[i];
    const b = det.box;
    const x1 = b[0], y1 = b[1], x2 = b[2], y2 = b[3];

    const vx1 = MIRROR ? (vw - x2) : x1;
    const vx2 = MIRROR ? (vw - x1) : x2;

    const rx1 = ox + vx1 * scale;
    const ry1 = oy + y1  * scale;
    const rwc = (vx2 - vx1) * scale;
    const rhc = (y2  - y1) * scale;

    overlayCtx!.strokeStyle = '#00ff88';
    overlayCtx!.fillStyle = 'rgba(0,0,0,0.5)';
    overlayCtx!.strokeRect(rx1, ry1, rwc, rhc);

    const label = `${det.label} ${(det.score * 100 + 0.5 | 0)}%`;
    const tw = overlayCtx!.measureText(label).width + 6;
    overlayCtx!.fillRect(rx1, Math.max(0, ry1 - 2), tw, 18);
    overlayCtx!.fillStyle = '#00ff88';
    overlayCtx!.fillText(label, rx1 + 3, ry1);

    people++;
  }

  const hud = `Persons: ${people}`;
  const hudW = overlayCtx!.measureText(hud).width + 12;
  overlayCtx!.fillStyle = 'rgba(0,0,0,0.6)';
  overlayCtx!.fillRect(10, 10, hudW, 24);
  overlayCtx!.fillStyle = '#fbff68';
  overlayCtx!.fillText(hud, 16, 14);

  if (channelEl) channelEl.textContent = people > 0 ? `AV1 (${people})` : 'AV1';
}

// ---------- Chargement ORT (WebGPU only) ----------
async function loadOrtWebGPU(): Promise<void> {
  const hasWebGPU = typeof navigator !== 'undefined' && !!(navigator as any).gpu;
  if (!hasWebGPU) {
    throw new Error('WebGPU non disponible. Active-le (Chrome/Edge/Firefox récent) et utilise HTTPS.');
  }

  const ortWebgpu = await import('onnxruntime-web/webgpu');

  // Certains artefacts wasm internes : fixer le CDN pour éviter des 404
  (ortWebgpu as any).env.wasm = (ortWebgpu as any).env.wasm || {};
  (ortWebgpu as any).env.wasm.wasmPaths =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

  (ortWebgpu as any).env.webgpu = (ortWebgpu as any).env.webgpu || {};
  (ortWebgpu as any).env.webgpu.powerPreference = 'high-performance';
  (ortWebgpu as any).env.webgpu.enableGraphCapture = true;

  // @ts-ignore
  ort = ortWebgpu;
}

async function createSessionWebGPU() {
  session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['webgpu'] as any
  });
  INPUT_NAME = session.inputNames[0];
  const meta: any = (session as any).inputMetadata?.[INPUT_NAME];
  const dims: number[] = meta?.dimensions ?? [];
  if (Array.isArray(dims) && dims.length === 4) {
    if (dims[1] === 3) INPUT_LAYOUT = 'NCHW';
    else if (dims[3] === 3) INPUT_LAYOUT = 'NHWC';
  }
  console.log('[ORT] EPs: webgpu only');
  console.log('[ORT] Input dims:', dims, 'Layout:', INPUT_LAYOUT);
  console.log('[ORT] Inputs:', session.inputNames);
  console.log('[ORT] Outputs:', session.outputNames);
}

let running = false;
let usingVFC = false;

async function initDetector() {
  await loadOrtWebGPU();
  await createSessionWebGPU();
  // Optional warmup — une frame noire pour compiler le graph
  const zero = new Float32Array(3 * SIZE);
  const warm = new ort.Tensor('float32', INPUT_LAYOUT === 'NCHW'
    ? zero : zero,
    INPUT_LAYOUT === 'NCHW'
      ? [1,3,INPUT_SIZE,INPUT_SIZE]
      : [1,INPUT_SIZE,INPUT_SIZE,3]
  );
  const feeds: Record<string, ortTypes.Tensor> = { [INPUT_NAME]: warm };
  try { await session!.run(feeds); } catch { /* silencieux */ }
}

// ---------- Prétraitement réutilisable ----------
function letterboxAndToTensor(vw:number, vh:number) {
  // Letterbox -> preprocessCanvas (INPUT_SIZE x INPUT_SIZE)
  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const nw = (vw * scale + 0.5) | 0;
  const nh = (vh * scale + 0.5) | 0;
  const dx = ((INPUT_SIZE - nw) / 2 + 0.5) | 0;
  const dy = ((INPUT_SIZE - nh) / 2 + 0.5) | 0;

  pctx.setTransform(1, 0, 0, 1, 0, 0);
  pctx.fillStyle = '#000';
  pctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  pctx.drawImage(videoEl!, 0, 0, vw, vh, dx, dy, nw, nh);

  const img = pctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data; // RGBA

  if (INPUT_LAYOUT === 'NCHW') {
    // planéifier en CHW
    // micro-unroll
    for (let i=0, p=0; i<SIZE; i++, p+=4) {
      const r = img[p] * (1/255);
      const g = img[p+1] * (1/255);
      const b = img[p+2] * (1/255);
      nchwBuf[i]        = r;
      nchwBuf[i + SIZE] = g;
      nchwBuf[i + 2*SIZE] = b;
    }
  } else {
    for (let i=0, p=0, o=0; i<SIZE; i++, p+=4, o+=3) {
      nhwcBuf[o]   = img[p]   * (1/255);
      nhwcBuf[o+1] = img[p+1] * (1/255);
      nhwcBuf[o+2] = img[p+2] * (1/255);
    }
  }

  return {
    tensor: new ort.Tensor(
      'float32',
      INPUT_LAYOUT === 'NCHW' ? nchwBuf : nhwcBuf,
      INPUT_LAYOUT === 'NCHW'
        ? [1, 3, INPUT_SIZE, INPUT_SIZE]
        : [1, INPUT_SIZE, INPUT_SIZE, 3]
    ),
    letter: { sx: scale, sy: scale, dw: dx, dh: dy, rw: vw, rh: vh } as Letterbox
  };
}

// ---------- Boucle d’inférence ----------

type RAFLike = (cb: FrameRequestCallback) => number;
let vfcHandle = 0;
let rafHandle = 0;

function scheduleNext(fn: () => void) {
  if (usingVFC && 'requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    vfcHandle = (videoEl as any).requestVideoFrameCallback(() => fn());
  } else {
    rafHandle = requestAnimationFrame(() => fn());
  }
}

function cancelScheduled() {
  if (vfcHandle && 'cancelVideoFrameCallback' in HTMLVideoElement.prototype) {
    (videoEl as any).cancelVideoFrameCallback(vfcHandle);
    vfcHandle = 0;
  }
  if (rafHandle) {
    cancelAnimationFrame(rafHandle);
    rafHandle = 0;
  }
}

async function inferOnce() {
  if (!session || videoEl!.readyState < 2) return;

  resizeOverlayToVideoIfNeeded();

  const vw = Math.max(1, videoEl!.videoWidth), vh = Math.max(1, videoEl!.videoHeight);
  const { tensor, letter } = letterboxAndToTensor(vw, vh);

  const feeds: Record<string, ortTypes.Tensor> = { [INPUT_NAME]: tensor };

  let results: Record<string, ortTypes.Tensor>;
  try {
    results = await session.run(feeds);
  } catch (e: any) {
    const msg = String(e?.message || e);

    const softmaxBadAxis =
      msg.includes('softmax only supports last axis for now') ||
      (msg.includes('Softmax') && msg.toLowerCase().includes('failed'));
    if (softmaxBadAxis) {
      throw new Error(
        'Le modèle utilise Softmax sur un axe non supporté en WebGPU. ' +
        'Patch l’ONNX (transpose -> Softmax(axis=-1) -> transpose inverse) puis réessaie.'
      );
    }

    if (msg.includes('Got: 3 Expected: 640') && msg.includes('Got: 640 Expected: 3')) {
      INPUT_LAYOUT = (INPUT_LAYOUT === 'NHWC') ? 'NCHW' : 'NHWC';
      console.warn('Layout auto-switch ->', INPUT_LAYOUT);
      return; // on retentera à la frame suivante
    }

    throw e;
  }

  const dets = decodeYoloV8(results, letter);
  drawDetections(dets);
}

async function loop() {
  if (!running) return;
  await inferOnce();
  if (running) scheduleNext(loop);
}

// ---------- API publique ----------
export async function startDetection() {
  if (!session) await initDetector();

  // Utiliser rVFC si disponible (collé aux frames vidéo)
  usingVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

  running = true;
  scheduleNext(loop);
}

export function stopDetection() {
  running = false;
  cancelScheduled();
  overlayCtx!.clearRect(0, 0, overlay!.width, overlay!.height);
}

// Expose pour index.html
;(window as any).startDetection = startDetection;
;(window as any).stopDetection = stopDetection;
