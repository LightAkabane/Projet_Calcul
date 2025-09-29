// main.ts

// ---------- Types ----------
import type * as ortTypes from 'onnxruntime-web';
// ---------- Utils YOLO ----------
const INPUT_SIZE = 640;

// Seuil final (après produit obj*cls). On peut le laisser raisonnable :
const SCORE_THRESH = 0.55;

// Rendez NMS un poil plus agressif (plus de suppression)
const NMS_IOU = 0.40;

// Limite globale (après NMS)
const MAX_DETECTIONS = 100;

// Gating AVANT NMS (indépendant)
// → ça coupe le bruit des cellules qui “cliquettent” juste un peu
const OBJ_THRESH = 0.45;      // objectness mini
const PERSON_THRESH = 0.45;   // prob. de la classe 'person' mini

// Top-K par tête (par stride) AVANT fusion globale
const TOPK_PER_HEAD = 400;    // tu peux descendre à 300/200 si besoin

// Taille mini (en pixels vidéo) pour écarter les micro-boîtes parasites
const MIN_SIDE_FRAC = 0.02;   // 2% du côté le plus court du flux vidéo

// ---------- ONNX Runtime (WASM 1.19.2) ----------
let ort: typeof import('onnxruntime-web');
let INPUT_LAYOUT: 'NCHW' | 'NHWC' = 'NHWC';
let INPUT_NAME = '';
const MIRROR = true; // ta <video> est en transform: scaleX(-1)

async function loadOrt(): Promise<'wasm'> {
  ort = await import('onnxruntime-web'); // assure onnxruntime-web@1.19.2
  return 'wasm';
}

// ---------- Chemins / options ----------
const MODEL_PATH = '/yolov8m.onnx';
const WASM_CDN_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

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
// Renvoie un accès get(ch, i) avec i = index aplati (y*W + x) si grille
type Accessor = {
  N: number;             // nombre de points (H*W ou N direct)
  C: number;             // nombre de canaux (>= 5)
  get: (c: number, i: number) => number;
};
function makeAccessor(t: ortTypes.Tensor): Accessor | null {
  const dims = t.dims;
  const data = t.data as Float32Array;

  // 4D: [1, H, W, C] ou [1, C, H, W] (ou sans le 1 si exotique)
  if (dims.length === 4) {
    const [d0,d1,d2,d3] = dims;
    // [1, H, W, C]
    if (d0 === 1 && d3 >= 5) {
      const H = d1, W = d2, C = d3, N = H * W;
      return {
        N, C,
        get: (ch, i) => {
          const x = i % W;
          const y = (i - x) / W;
          // NHWC: index = y*W*C + x*C + ch
          return data[(y * W + x) * C + ch];
        }
      };
    }
    // [1, C, H, W]
    if (d0 === 1 && d1 >= 5) {
      const C = d1, H = d2, W = d3, N = H * W;
      return {
        N, C,
        get: (ch, i) => {
          const x = i % W;
          const y = (i - x) / W;
          // NCHW: index = ch*H*W + y*W + x
          return data[ch * H * W + y * W + x];
        }
      };
    }
  }

  // 3D variantes
  if (dims.length === 3) {
    const [a, b, c] = dims;
    // [1,84,N]
    if (a === 1 && b >= 5) {
      const C = b, N = c;
      return { N, C, get: (ch,i) => data[ch*N + i] };
    }
    // [1,N,84]
    if (a === 1 && c >= 5) {
      const N = b, C = c;
      return { N, C, get: (ch,i) => data[i*C + ch] };
    }
    // [84,1,N]
    if (b === 1 && a >= 5) {
      const C = a, N = c;
      return { N, C, get: (ch,i) => data[ch*N + i] };
    }
    // [N,1,84]
    if (b === 1 && c >= 5) {
      const N = a, C = c;
      return { N, C, get: (ch,i) => data[i*C + ch] };
    }
  }

  // 2D variantes
  if (dims.length === 2) {
    const [a, b] = dims;
    // [C, N]
    if (a >= 5) {
      const C = a, N = b;
      return { N, C, get: (ch,i) => data[ch*N + i] };
    }
    // [N, C]
    if (b >= 5) {
      const N = a, C = b;
      return { N, C, get: (ch,i) => data[i*C + ch] };
    }
  }

  console.warn('makeAccessor: forme inattendue ignorée:', dims);
  return null; // on ignore cette tête
}

// --- Décodage YOLOv8 (anchor-free) pour têtes brutes (H*W*C) ---
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

    // stride via taille de grille
    const g = Math.max(1, Math.round(Math.sqrt(N)));   // ex: 80, 40, 20
    const stride = INPUT_SIZE / g;

    const numClasses = Math.max(0, C - 5);
    if (numClasses < 1) continue;

    // Collecte locale pour faire un Top-K *par tête*
    const headBoxes:number[][] = [];
    const headScores:number[] = [];

    for (let i=0; i<N; i++) {
      const cx = i % g;
      const cy = (i - cx) / g;

      // logits
      const tx = get(0,i),  ty = get(1,i),  tw = get(2,i),  th = get(3,i);
      const to = get(4,i);               // objectness
      const tPerson = get(5 + 0, i);     // classe 0 = person (COCO)

      const pObj = 1/(1+Math.exp(-to));
      const pCls = 1/(1+Math.exp(-tPerson));

      // Gating indépendant (coupe beaucoup de bruit)
      if (pObj < OBJ_THRESH || pCls < PERSON_THRESH) continue;

      // Décodage YOLOv8 (anchor-free)
      const x = ((1/(1+Math.exp(-tx)) * 2 - 0.5) + cx) * stride;
      const y = ((1/(1+Math.exp(-ty)) * 2 - 0.5) + cy) * stride;
      const w =  ( (1/(1+Math.exp(-tw)) * 2) ** 2 ) * stride;
      const h =  ( (1/(1+Math.exp(-th)) * 2) ** 2 ) * stride;

      // Taille mini (évite les boîtes parasites)
      if (w < minSidePx || h < minSidePx) continue;

      const conf = pObj * pCls;
      if (conf < SCORE_THRESH) continue;

      // xywh -> xyxy dans l’espace 640x640
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

    // --- Top-K par tête avant fusion globale ---
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

  // NMS class-agnostic (on ne garde que 'person')
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

// ---------- ORT session + boucle ----------
let session: ortTypes.InferenceSession | null = null;
let running = false;

async function initDetector() {
  await loadOrt(); // WASM

  if ((ort as any)?.env?.wasm) {
    (ort as any).env.wasm.wasmPaths = WASM_CDN_BASE; // CDN piné 1.19.2
    (ort as any).env.wasm.simd = true;
    (ort as any).env.wasm.numThreads = 1;   // dev non-COI => 1 thread
    (ort as any).env.wasm.proxy = false;    // pas de worker/proxy
    (ort as any).env.wasm.wasmFile = 'ort-wasm-simd.wasm'; // non-threaded
  }

  session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['wasm'] as any
  });

  // Déduire le layout si possible (NCHW/NHWC)
  INPUT_NAME = session.inputNames[0];
  const meta: any = (session as any).inputMetadata?.[INPUT_NAME];
  const dims: number[] = meta?.dimensions ?? [];
  if (Array.isArray(dims) && dims.length === 4) {
    if (dims[1] === 3) INPUT_LAYOUT = 'NCHW';
    else if (dims[3] === 3) INPUT_LAYOUT = 'NHWC';
  }

  console.log('Input dims:', dims, 'Layout:', INPUT_LAYOUT);
  console.log('ORT inputs:', session.inputNames);
  console.log('ORT outputs:', session.outputNames);
}

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
    if (msg.includes('Got: 3 Expected: 640') && msg.includes('Got: 640 Expected: 3')) {
      INPUT_LAYOUT = (INPUT_LAYOUT === 'NHWC') ? 'NCHW' : 'NHWC';
      console.warn('Layout auto-switch ->', INPUT_LAYOUT);
      requestAnimationFrame(inferLoop);
      return;
    }
    throw e;
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
