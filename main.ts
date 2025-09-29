// main.ts — WebGPU-first + fallback WASM, decode YOLOv8 concat (8400)

// ---------- Types ----------
import type * as ortTypes from 'onnxruntime-web';

// ---------- Hyperparams ----------
const INPUT_SIZE = 640;
const NMS_IOU = 0.40;
const MAX_DETECTIONS = 100;
const TOPK_PER_HEAD = 400;
const SCORE_THRESH = 0.30;   // 0.55 -> 0.30
const OBJ_THRESH   = 0.25;   // 0.45 -> 0.25
const PERSON_THRESH= 0.25;   // 0.45 -> 0.25
const MIN_SIDE_FRAC= 0.015;  // 0.02  -> 0.015


// ---------- ONNX Runtime ----------
let ort: typeof import('onnxruntime-web');
let session: ortTypes.InferenceSession | null = null;

let INPUT_LAYOUT: 'NCHW' | 'NHWC' = 'NHWC';
let INPUT_NAME = '';
const MIRROR = true; // ta <video> est en transform: scaleX(-1)

// ---------- Chemins / options ----------
const MODEL_PATH = '/yolov8n.onnx'; // mets ici ton .onnx (patché ou non)

// ---------- DOM ----------
const videoEl = document.getElementById('cam') as HTMLVideoElement | null;
const overlay = document.getElementById('overlay') as HTMLCanvasElement | null;
const channelEl = document.getElementById('channel') as HTMLDivElement | null;

if (!videoEl) throw new Error('#cam introuvable (video)');
if (!overlay) throw new Error('#overlay introuvable (canvas)');

const overlayCtx = overlay.getContext('2d');
if (!overlayCtx) throw new Error('Context 2D indisponible pour #overlay');

// DPI-safe resize du canvas
function resizeOverlayToVideo() {
  const rect = overlay!.getBoundingClientRect(); // taille CSS
  const dpr = window.devicePixelRatio || 1;
  overlay!.style.width = '100%';
  overlay!.style.height = '100%';
  const w = Math.max(1, Math.floor(rect.width * dpr));
  const h = Math.max(1, Math.floor(rect.height * dpr));
  if (overlay!.width !== w || overlay!.height !== h) {
    overlay!.width = w;
    overlay!.height = h;
    overlayCtx!.setTransform(dpr, 0, 0, dpr, 0, 0); // dessiner en unités CSS
  }
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

function iou(a: number[], b: number[]): number {
  const [ax1, ay1, ax2, ay2] = a, [bx1, by1, bx2, by2] = b;
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = iw * ih;
  const areaA = (ax2 - ax1) * (ay2 - ay1);
  const areaB = (bx2 - bx1) * (ay2 - by1);
  return inter / (areaA + areaB - inter + 1e-6);
}

function nms(boxes: number[][], scores: number[], iouTh = NMS_IOU, limit = MAX_DETECTIONS): number[] {
  const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).map(([, i]) => i);
  const keep: number[] = [];
  for (const i of order) {
    if (keep.length >= limit) break;
    let ok = true;
    for (const j of keep) {
      if (iou(boxes[i], boxes[j]) > iouTh) { ok = false; break; }
    }
    if (ok) keep.push(i);
  }
  return keep;
}

// --- Accessor générique pour têtes 2D/3D/4D ---
type Accessor = {
  N: number;  // points (H*W ou N)
  C: number;  // >= 5
  get: (c: number, i: number) => number;
};

function makeAccessor(t: ortTypes.Tensor): Accessor | null {
  const dims = t.dims;
  const data = t.data as Float32Array;

  // 4D: [1, H, W, C] ou [1, C, H, W]
  if (dims.length === 4) {
    const [d0,d1,d2,d3] = dims;
    if (d0 === 1 && d3 >= 5) {
      const H = d1, W = d2, C = d3, N = H * W;
      return {
        N, C,
        get: (ch, i) => {
          const x = i % W;
          const y = (i - x) / W;
          return data[(y * W + x) * C + ch];
        }
      };
    }
    if (d0 === 1 && d1 >= 5) {
      const C = d1, H = d2, W = d3, N = H * W;
      return {
        N, C,
        get: (ch, i) => {
          const x = i % W;
          const y = (i - x) / W;
          return data[ch * H * W + y * W + x];
        }
      };
    }
  }

  // 3D
  if (dims.length === 3) {
    const [a, b, c] = dims;
    if (a === 1 && b >= 5) {
      const C = b, N = c;
      return { N, C, get: (ch,i) => data[ch*N + i] };
    }
    if (a === 1 && c >= 5) {
      const N = b, C = c;
      return { N, C, get: (ch,i) => data[i*C + ch] };
    }
    if (b === 1 && a >= 5) {
      const C = a, N = c;
      return { N, C, get: (ch,i) => data[ch*N + i] };
    }
    if (b === 1 && c >= 5) {
      const N = a, C = c;
      return { N, C, get: (ch,i) => data[i*C + ch] };
    }
  }

  // 2D
  if (dims.length === 2) {
    const [a, b] = dims;
    if (a >= 5) {
      const C = a, N = b;
      return { N, C, get: (ch,i) => data[ch*N + i] };
    }
    if (b >= 5) {
      const N = a, C = b;
      return { N, C, get: (ch,i) => data[i*C + ch] };
    }
  }

  console.warn('makeAccessor: forme inattendue ignorée:', dims);
  return null;
}

// ---------- Décodage YOLOv8 (anchor-free) ----------

// tailles des têtes pour INPUT_SIZE (v8: strides 8/16/32)
const HEAD_GS = [INPUT_SIZE / 8, INPUT_SIZE / 16, INPUT_SIZE / 32].map(v => Math.round(v)); // [80,40,20] si 640
const HEAD_COUNTS = HEAD_GS.map(g => g * g); // [6400,1600,400]
const HEAD_OFFSETS = [0, HEAD_COUNTS[0], HEAD_COUNTS[0] + HEAD_COUNTS[1]]; // [0,6400,8000]
const SUM_HEADS = HEAD_COUNTS.reduce((a,b)=>a+b,0); // 8400 pour 640

function gridInfoForIndex(i: number) {
  if (i < HEAD_OFFSETS[1]) {
    const g = HEAD_GS[0], base = HEAD_OFFSETS[0];
    const ii = i - base;
    const cx = ii % g, cy = (ii - cx) / g;
    return { g, cx, cy, stride: INPUT_SIZE / g };
  } else if (i < HEAD_OFFSETS[2]) {
    const g = HEAD_GS[1], base = HEAD_OFFSETS[1];
    const ii = i - base;
    const cx = ii % g, cy = (ii - cx) / g;
    return { g, cx, cy, stride: INPUT_SIZE / g };
  } else {
    const g = HEAD_GS[2], base = HEAD_OFFSETS[2];
    const ii = i - base;
    const cx = ii % g, cy = (ii - cx) / g;
    return { g, cx, cy, stride: INPUT_SIZE / g };
  }
}

function decodeYoloV8(
  results: Record<string, ortTypes.Tensor>,
  letter: { sx:number, sy:number, dw:number, dh:number, rw:number, rh:number }
) {
  const outs = Object.values(results);

  const allBoxes:number[][] = [];
  const allScores:number[] = [];

  // seuil taille mini en pixels vidéo
  const minSidePx = Math.max(8, Math.min(letter.rw, letter.rh) * MIN_SIDE_FRAC);

  for (const t of outs) {
    const acc = makeAccessor(t);
    if (!acc) continue;
    const { N, C, get } = acc;

    const numClasses = Math.max(0, C - 5);
    if (numClasses < 1) continue;

    const headBoxes:number[][] = [];
    const headScores:number[] = [];

    for (let i=0; i<N; i++) {
      // --- grilles/stride corrects (concat 8400) ---
      let g:number, cx:number, cy:number, stride:number;
      if (N === SUM_HEADS) {
        const gi = gridInfoForIndex(i);
        g = gi.g; cx = gi.cx; cy = gi.cy; stride = gi.stride;
      } else {
        // fallback si têtes séparées: approx selon sqrt(N)
        g = Math.max(1, Math.round(Math.sqrt(N)));
        stride = INPUT_SIZE / g;
        cx = i % g;
        cy = (i - cx) / g;
      }

      // logits
      const tx = get(0,i),  ty = get(1,i),  tw = get(2,i),  th = get(3,i);
      const to = get(4,i);               // objectness
      const tPerson = get(5 + 0, i);     // classe 0 = person (COCO)

      const pObj = sigmoid(to);
      const pCls = sigmoid(tPerson);

      // gating indépendant
      if (pObj < OBJ_THRESH || pCls < PERSON_THRESH) continue;

      // Décodage YOLOv8 (anchor-free)
      const x = ((sigmoid(tx) * 2 - 0.5) + cx) * stride;
      const y = ((sigmoid(ty) * 2 - 0.5) + cy) * stride;
      const w =  ( (sigmoid(tw) * 2) ** 2 ) * stride;
      const h =  ( (sigmoid(th) * 2) ** 2 ) * stride;

      if (w < minSidePx || h < minSidePx) continue;

      const conf = pObj * pCls;
      if (conf < SCORE_THRESH) continue;

      // xywh -> xyxy (espace 640)
      const x1l = x - w/2, y1l = y - h/2, x2l = x + w/2, y2l = y + h/2;

      // dé-letterbox -> pixels vidéo
      const x1 = Math.max(0, (x1l - letter.dw) / letter.sx);
      const y1 = Math.max(0, (y1l - letter.dh) / letter.sy);
      const x2 = Math.min(letter.rw, (x2l - letter.dw) / letter.sx);
      const y2 = Math.min(letter.rh, (y2l - letter.dh) / letter.sy);

      if (!isFinite(x1) || !isFinite(y1) || !isFinite(x2) || !isFinite(y2)) continue;
      if (x2 <= x1 || y2 <= y1) continue;

      headBoxes.push([x1,y1,x2,y2]);
      headScores.push(conf);
    }

    // Top-K par tête avant fusion globale
    if (headBoxes.length > TOPK_PER_HEAD) {
      const order = headScores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).slice(0, TOPK_PER_HEAD).map(p=>p[1]);
      for (const idx of order) {
        allBoxes.push(headBoxes[idx]);
        allScores.push(headScores[idx]);
      }
    } else {
      allBoxes.push(...headBoxes);
      allScores.push(...headScores);
    }
  }

  // NMS class-agnostic (person uniquement)
  const keep = nms(allBoxes, allScores, NMS_IOU, MAX_DETECTIONS);
  return keep.map(k => ({ box: allBoxes[k], score: allScores[k], cls: 0, label: 'person' }));
}

// ---------- Dessin ----------
function drawDetections(dets: { box:number[], score:number, cls:number, label:string }[]) {
  const vw = Math.max(1, videoEl!.videoWidth);
  const vh = Math.max(1, videoEl!.videoHeight);

  const rect = overlay!.getBoundingClientRect();
  const cw = rect.width;
  const ch = rect.height;

  // mapping object-fit: cover
  const scale = Math.max(cw / vw, ch / vh);
  const rw = vw * scale;
  const rh = vh * scale;
  const ox = (cw - rw) / 2;
  const oy = (ch - rh) / 2;

  overlayCtx!.clearRect(0, 0, cw, ch);
  overlayCtx!.lineWidth = 2;
  overlayCtx!.font = '16px monospace';
  overlayCtx!.textBaseline = 'top';

  let people = 0;

  for (const det of dets) {
    const [x1, y1, x2, y2] = det.box;

    // miroir éventuel (axe vertical)
    const vx1 = MIRROR ? (vw - x2) : x1;
    const vx2 = MIRROR ? (vw - x1) : x2;

    const rx1 = ox + vx1 * scale;
    const ry1 = oy + y1  * scale;
    const rwc = (vx2 - vx1) * scale;
    const rhc = (y2  - y1) * scale;

    overlayCtx!.strokeStyle = '#00ff88';
    overlayCtx!.fillStyle = 'rgba(0,0,0,0.5)';
    overlayCtx!.strokeRect(rx1, ry1, rwc, rhc);

    const label = `${det.label} ${(det.score * 100).toFixed(0)}%`;
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

// ---------- Chargement ORT + sessions ----------
async function loadOrt(): Promise<void> {
  const hasWebGPU = typeof navigator !== 'undefined' && !!(navigator as any).gpu;

  if (hasWebGPU) {
    // Import WebGPU en priorité
    const ortWebgpu = await import('onnxruntime-web/webgpu');
    // Même en WebGPU, ORT s’appuie sur des artefacts .wasm -> pin CDN
    (ortWebgpu as any).env.wasm = (ortWebgpu as any).env.wasm || {};
    (ortWebgpu as any).env.wasm.wasmPaths =
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

    (ortWebgpu as any).env.webgpu = (ortWebgpu as any).env.webgpu || {};
    (ortWebgpu as any).env.webgpu.powerPreference = 'high-performance';
    (ortWebgpu as any).env.webgpu.enableGraphCapture = true;

    // @ts-ignore
    ort = ortWebgpu;
    return;
  }

  // Fallback WASM
  const ortWasm = await import('onnxruntime-web');
  (ortWasm as any).env.wasm = (ortWasm as any).env.wasm || {};
  (ortWasm as any).env.wasm.wasmPaths =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
  (ortWasm as any).env.wasm.simd = true;
  (ortWasm as any).env.wasm.numThreads = 1; // si COOP/COEP: tu peux monter et charger le .threaded
  (ortWasm as any).env.wasm.proxy = false;
  (ortWasm as any).env.wasm.wasmFile = 'ort-wasm-simd.wasm';

  // @ts-ignore
  ort = ortWasm;
}

async function createSessionWithEPs(eps: ('webgpu'|'wasm')[]) {
  session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: eps as any
  });
  INPUT_NAME = session.inputNames[0];
  const meta: any = (session as any).inputMetadata?.[INPUT_NAME];
  const dims: number[] = meta?.dimensions ?? [];
  if (Array.isArray(dims) && dims.length === 4) {
    if (dims[1] === 3) INPUT_LAYOUT = 'NCHW';
    else if (dims[3] === 3) INPUT_LAYOUT = 'NHWC';
  }
  console.log('[ORT] EPs:', eps);
  console.log('[ORT] Input dims:', dims, 'Layout:', INPUT_LAYOUT);
  console.log('[ORT] Inputs:', session.inputNames);
  console.log('[ORT] Outputs:', session.outputNames);
}

let running = false;

async function initDetector() {
  await loadOrt();
  // On tente WebGPU puis WASM en secours
  await createSessionWithEPs(['webgpu','wasm']);
}

// ---------- Boucle d'inférence ----------
async function inferLoop() {
  if (!session || videoEl!.readyState < 2) { requestAnimationFrame(inferLoop); return; }
  resizeOverlayToVideo();

  // --- Prétraitement (letterbox + tensor) ---
  const vw = Math.max(1, videoEl!.videoWidth), vh = Math.max(1, videoEl!.videoHeight);
  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const nw = Math.round(vw * scale), nh = Math.round(vh * scale);
  const dx = Math.floor((INPUT_SIZE - nw) / 2);
  const dy = Math.floor((INPUT_SIZE - nh) / 2);

  const tmp = document.createElement('canvas');
  tmp.width = INPUT_SIZE; tmp.height = INPUT_SIZE;
  const tctx = tmp.getContext('2d')!;
  tctx.fillStyle = '#000';
  tctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  tctx.drawImage(videoEl!, 0, 0, vw, vh, dx, dy, nw, nh);

  const { data } = tctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE); // RGBA
  const size = INPUT_SIZE * INPUT_SIZE;

  let tensor: ortTypes.Tensor;
  if (INPUT_LAYOUT === 'NCHW') {
    const chw = new Float32Array(3 * size);
    for (let i = 0; i < size; i++) {
      chw[i]           = data[i * 4]     / 255;
      chw[i + size]    = data[i * 4 + 1] / 255;
      chw[i + size*2]  = data[i * 4 + 2] / 255;
    }
    tensor = new ort.Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  } else {
    const nhwc = new Float32Array(3 * size);
    for (let i = 0; i < size; i++) {
      const o = i * 3;
      nhwc[o]   = data[i * 4]     / 255;
      nhwc[o+1] = data[i * 4 + 1] / 255;
      nhwc[o+2] = data[i * 4 + 2] / 255;
    }
    tensor = new ort.Tensor('float32', nhwc, [1, INPUT_SIZE, INPUT_SIZE, 3]);
  }

  const letterbox = { sx: scale, sy: scale, dw: dx, dh: dy, rw: vw, rh: vh };

  const feeds: Record<string, ortTypes.Tensor> = {};
  feeds[INPUT_NAME] = tensor;

  let results: Record<string, ortTypes.Tensor>;
  try {
    results = await session.run(feeds);
  } catch (e: any) {
    const msg = String(e?.message || e);

    // Softmax non supporté (axe ≠ dernier) côté WebGPU -> recrée en WASM only
    const softmaxBadAxis =
      msg.includes('softmax only supports last axis for now') ||
      (msg.includes('Softmax') && msg.toLowerCase().includes('failed'));

    if (softmaxBadAxis) {
      console.warn('[ORT] Softmax not-supported on WebGPU -> Recreate session in WASM only');
      await createSessionWithEPs(['wasm']);
      results = await session!.run(feeds);
    } else if (msg.includes('Got: 3 Expected: 640') && msg.includes('Got: 640 Expected: 3')) {
      INPUT_LAYOUT = (INPUT_LAYOUT === 'NHWC') ? 'NCHW' : 'NHWC';
      console.warn('Layout auto-switch ->', INPUT_LAYOUT);
      requestAnimationFrame(inferLoop);
      return;
    } else {
      throw e;
    }
  }

  const dets = decodeYoloV8(results, letterbox);
  drawDetections(dets);

  if (running) requestAnimationFrame(inferLoop);
}

// ---------- API publique ----------
export async function startDetection() {
  if (!session) await initDetector();
  running = true;
  inferLoop();
}
export function stopDetection() {
  running = false;
  overlayCtx!.clearRect(0, 0, overlay!.width, overlay!.height);
}

// Expose pour index.html
;(window as any).startDetection = startDetection;
;(window as any).stopDetection = stopDetection;
