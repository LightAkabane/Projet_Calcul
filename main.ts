// main.ts — YOLOv8 + Tracker ReID pour comptage unique (entrée fixe, WebGPU-first, perfs max)
// Version optimisée avec état de post-traitement encapsulé

import type * as ortTypes from 'onnxruntime-web';
import { PersonTracker, type Detection } from './tracker';

// ---------- Hyperparams YOLO ----------
const NMS_IOU = 0.30;          // ← plus strict
const MAX_DETECTIONS = 30;     // ← moins de boîtes finales
const TOPK_PER_HEAD = 150;     // ← on coupe l’excès tôt
const SCORE_THRESH = 0.35;     // ← filtre bruit
const OBJ_THRESH   = 0.40;     // ← filtre bruit
const PERSON_THRESH= 0.40;     // ← filtre bruit
const MIN_SIDE_FRAC= 0.025;
const DEFAULT_INPUT_SIZE = 640;
// --- tailles réseau dynamiques (fallback uniquement si le modèle est réellement dynamique) ---
const MAX_LONG_SIDE = 960; // pas utilisé si le modèle est fixe
const STRIDE = 32;

let ort: typeof import('onnxruntime-web');
let session: ortTypes.InferenceSession | null = null;
let tracker: PersonTracker | null = null;
let yoloProcessor: YoloV8Processor | null = null; // <<< OPTIMISATION: Instance de processeur

let INPUT_LAYOUT: 'NCHW' | 'NHWC' = 'NHWC'; // sera corrigé après lecture des métadonnées ONNX
let INPUT_NAME = '';
const MIRROR = true;
const MODEL_PATH = '/yolov8m.onnx'; // ONNX exporté (ici supposé "fixe" HxW=640x640, sinon dynamique)

// ---------- DOM ----------
const videoEl = document.getElementById('cam') as HTMLVideoElement | null;
const overlay = document.getElementById('overlay') as HTMLCanvasElement | null;
const channelEl = document.getElementById('channel') as HTMLDivElement | null;

if (!videoEl) throw new Error('#cam introuvable');
if (!overlay) throw new Error('#overlay introuvable');

const overlayCtx = overlay.getContext('2d');
if (!overlayCtx) throw new Error('Context 2D indisponible');

// ---------- Canvases réutilisés ----------
const preprocessCanvas = document.createElement('canvas');
const pctx = preprocessCanvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D;

// --- entrée réseau (fixe si le modèle l’est) ---
let netW = 640, netH = 640; // mis à jour à l’init
let FIXED_W: number | null = null;
let FIXED_H: number | null = null;

let nhwcBuf: Float32Array | null = null;
let nchwBuf: Float32Array | null = null;

// ---------- Utils (Niveau Module) ----------
// Ces fonctions sont "pures" (stateless) et restent au niveau du module

function toMultipleOf(v: number, m: number) {
  return Math.max(m, (Math.round(v / m) * m) | 0);
}
function chooseNetSize(vw: number, vh: number) {
  const scale = Math.min(MAX_LONG_SIDE / vw, MAX_LONG_SIDE / vh, 1);
  let w = Math.max(1, Math.floor(vw * scale));
  let h = Math.max(1, Math.floor(vh * scale));
  w = toMultipleOf(w, STRIDE);
  h = toMultipleOf(h, STRIDE);
  return { w: Math.max(STRIDE, w), h: Math.max(STRIDE, h) };
}
function ensurePreprocessSize(w: number, h: number) {
  if (preprocessCanvas.width !== w || preprocessCanvas.height !== h) {
    preprocessCanvas.width = w;
    preprocessCanvas.height = h;
    nhwcBuf = new Float32Array(3 * w * h);
    nchwBuf = new Float32Array(3 * w * h);
  }
}

let lastDpr = 0;
let lastRectW = 0;
let lastRectH = 0;

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

function iou(a: number[], b: number[]): number {
  const [ax1, ay1, ax2, ay2] = a, [bx1, by1, bx2, by2] = b;
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
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

function weightedBoxesFusion(
  boxes: number[][],
  scores: number[],
  iouThr = 0.55,
  limit = MAX_DETECTIONS
): { boxes: number[][]; scores: number[] } {
  if (boxes.length === 0) return { boxes: [], scores: [] };

  const order = scores.map((s,i)=>[s,i] as const).sort((a,b)=>b[0]-a[0]).map(([,i])=>i);
  const used = new Uint8Array(boxes.length);
  const fusedBoxes: number[][] = [];
  const fusedScores: number[] = [];

  for (let oi = 0; oi < order.length; oi++) {
    const i = order[oi];
    if (used[i]) continue;

    let sumW = 0;
    let x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    let bestScore = scores[i];

    for (let oj = oi; oj < order.length; oj++) {
      const j = order[oj];
      if (used[j]) continue;
      if (iou(boxes[i], boxes[j]) >= iouThr) {
        const w = scores[j];
        sumW += w;
        x1 += boxes[j][0] * w;
        y1 += boxes[j][1] * w;
        x2 += boxes[j][2] * w;
        y2 += boxes[j][3] * w;
        if (scores[j] > bestScore) bestScore = scores[j];
        used[j] = 1;
      }
    }

    if (sumW > 0) {
      fusedBoxes.push([x1/sumW, y1/sumW, x2/sumW, y2/sumW]);
      fusedScores.push(bestScore);
      if (fusedBoxes.length >= limit) break;
    }
  }

  return { boxes: fusedBoxes, scores: fusedScores };
}

// === Accessor (on expose N et C ; H/W ne sont fiables que pour certains exports)
type Accessor = { N: number; C: number; get: (c: number, i: number) => number; };

function makeAccessor(t: ortTypes.Tensor): Accessor | null {
  const dims = t.dims;
  const data = t.data as Float32Array;

  // Cas commun Ultralytics: [1, 84, N]
  if (dims.length === 3) {
    const [a,b,c] = dims;
    if (a === 1 && b >= 5) { // [1, C, N]
      const C=b, N=c;
      return { N, C, get: (ch,i)=>data[ch*N+i] };
    }
    if (a === 1 && c >= 5) { // [1, N, C]
      const N=b, C=c;
      return { N, C, get: (ch,i)=>data[i*C+ch] };
    }
    if (b === 1 && a >= 5) { // [C,1,N]
      const C=a, N=c;
      return { N, C, get: (ch,i)=>data[ch*N+i] };
    }
    if (b === 1 && c >= 5) { // [N,1,C]
      const N=a, C=c;
      return { N, C, get: (ch,i)=>data[i*C+ch] };
    }
  }

  // 4D (quelques exports): [1, H, W, C] ou [1, C, H, W]
  if (dims.length === 4) {
    const [d0,d1,d2,d3] = dims;
    if (d0 === 1 && d3 >= 5) { // NHWC
      const H = d1, W = d2, C = d3, N = H * W;
      return { N, C, get: (ch,i) => { const x=i%W; const y=(i-x)/W; return data[(y*W+x)*C+ch]; } };
    }
    if (d0 === 1 && d1 >= 5) { // NCHW
      const C = d1, H = d2, W = d3, N = H * W;
      return { N, C, get: (ch,i) => { const x=i%W; const y=(i-x)/W; return data[ch*H*W + y*W + x]; } };
    }
  }

  // 2D fallback: [C,N] ou [N,C]
  if (dims.length === 2) {
    const [a,b] = dims;
    if (a >= 5) { // [C,N]
      const C=a, N=b;
      return { N, C, get: (ch,i)=>data[ch*N+i] };
    }
    if (b >= 5) { // [N,C]
      const N=a, C=b;
      return { N, C, get: (ch,i)=>data[i*C+ch] };
    }
  }

  console.warn('makeAccessor: forme inattendue ignorée:', dims);
  return null;
}

// === Lettre/resize info ===
type Letterbox = { sx:number, sy:number, dw:number, dh:number, rw:number, rh:number, inw:number, inh:number };

// Helpers têtes (dynamiques): strides 8/16/32
type HeadSpec = { gw:number; gh:number; stride:number; count:number; offset:number; };
function buildHeadSpecs(inw:number, inh:number): HeadSpec[] {
  const strides = [8,16,32];
  let offset = 0;
  const specs: HeadSpec[] = [];
  for (const s of strides) {
    const gw = Math.floor(inw / s);
    const gh = Math.floor(inh / s);
    const count = Math.max(0, gw * gh);
    specs.push({ gw, gh, stride: s, count, offset });
    offset += count;
  }
  return specs;
}

// Map index plat -> (cx,cy,stride) selon têtes
function indexToGrid(i:number, specs:HeadSpec[]): { cx:number; cy:number; stride:number } {
  for (let h=0; h<specs.length; h++) {
    const sp = specs[h];
    if (i < sp.offset + sp.count) {
      const local = i - sp.offset;
      const cx = local % sp.gw;
      const cy = (local - cx) / sp.gw;
      return { cx, cy, stride: sp.stride };
    }
  }
  // fallback (devrait pas arriver)
  return { cx: 0, cy: 0, stride: 8 };
}
function suppressByCenter(
  boxes: number[][],
  scores: number[],
  centerFrac = 0.40 // rayon = 40% de la plus petite dimension
): number[] {
  if (boxes.length === 0) return [];
  const order = scores.map((s,i)=>[s,i] as const).sort((a,b)=>b[0]-a[0]).map(([,i])=>i);
  const keep: number[] = [];
  const taken = new Uint8Array(boxes.length);

  for (let oi = 0; oi < order.length; oi++) {
    const i = order[oi];
    if (taken[i]) continue;
    keep.push(i);
    taken[i] = 1;

    const bi = boxes[i];
    const cxi = 0.5*(bi[0]+bi[2]);
    const cyi = 0.5*(bi[1]+bi[3]);
    const wi = Math.max(1e-3, bi[2]-bi[0]);
    const hi = Math.max(1e-3, bi[3]-bi[1]);
    const r = centerFrac * Math.min(wi, hi);

    for (let oj = oi+1; oj < order.length; oj++) {
      const j = order[oj];
      if (taken[j]) continue;
      const bj = boxes[j];
      const cxj = 0.5*(bj[0]+bj[2]);
      const cyj = 0.5*(bj[1]+bj[3]);
      const dx = cxi - cxj;
      const dy = cyi - cyj;
      if ((dx*dx + dy*dy) <= (r*r)) {
        // même centre → on garde la meilleure (déjà i), on marque j comme supprimée
        taken[j] = 1;
      }
    }
  }
  return keep;
}

function centerDist2(a:number[], b:number[]) {
  const acx = 0.5*(a[0]+a[2]), acy = 0.5*(a[1]+a[3]);
  const bcx = 0.5*(b[0]+b[2]), bcy = 0.5*(b[1]+b[3]);
  const dx = acx - bcx, dy = acy - bcy;
  return dx*dx + dy*dy;
}


// =======================================================
// === CLASSE DE POST-TRAITEMENT YOLO (Optimisation) ===
// =======================================================
/**
 * Encapsule la logique de décodage YOLO et gère l'état
 * interne pour le lissage temporel.
 */
class YoloV8Processor {
  // État interne pour le lissage temporel
  private _prevBoxes: number[][] = [];
  private _prevScores: number[] = [];
  
  // Spécifications des têtes du modèle (calculées une fois)
  private headSpecs: HeadSpec[];
  private inw: number;
  private inh: number;

  constructor(inw: number, inh: number) {
    this.inw = inw;
    this.inh = inh;
    // On présuppose une taille fixe, on calcule les specs 1 seule fois
    this.headSpecs = buildHeadSpecs(inw, inh);
  }

  /** Réinitialise l'état du lissage temporel */
  public reset() {
    this._prevBoxes = [];
    this._prevScores = [];
  }

  /**
   * Lissage + dédoublonnage temporel
   * OPTIMISATION: Gère son propre état (this._prevBoxes)
   */
  private temporalSmoothAndDedup(
    boxes: number[][],
    scores: number[],
    {
      iouGate = 0.35,        // assez permissif (mouvement = IoU baisse)
      centerFrac = 0.60,     // rayon de match si IoU trop faible
      alpha = 0.65,          // lissage exp. vers la position précédente
      limit = MAX_DETECTIONS // plafond
    } = {}
  ): { boxes: number[][]; scores: number[] } {
    if (!boxes.length) { 
      this._prevBoxes = []; 
      this._prevScores = []; 
      return { boxes, scores }; 
    }

    // Ordonne par score (haut -> bas)
    const order = scores.map((s,i)=>[s,i] as const).sort((a,b)=>b[0]-a[0]).map(([,i])=>i);
    const usedPrev = new Uint8Array(this._prevBoxes.length);
    const outB: number[][] = [];
    const outS: number[] = [];

    for (const i of order) {
      const b = boxes[i];
      const s = scores[i];

      // match meilleur précédent par IoU (sinon centre)
      let bestJ = -1, bestIoU = 0;
      for (let j=0; j<this._prevBoxes.length; j++) {
        if (usedPrev[j]) continue;
        const pb = this._prevBoxes[j];
        let ok = false;
        const iouv = iou(b, pb); // Utilise l'util globale
        if (iouv >= iouGate) ok = true;
        else {
          const wi = Math.max(1e-3, b[2]-b[0]);
          const hi = Math.max(1e-3, b[3]-b[1]);
          const r2 = (centerFrac* Math.min(wi,hi))**2;
          if (centerDist2(b, pb) <= r2) ok = true; // Utilise l'util globale
        }
        if (ok && iouv > bestIoU) { bestIoU = iouv; bestJ = j; }
      }

      if (bestJ >= 0) {
        // lissage vers la box précédente
        const pb = this._prevBoxes[bestJ];
        const sm = [
          alpha*pb[0] + (1-alpha)*b[0],
          alpha*pb[1] + (1-alpha)*b[1],
          alpha*pb[2] + (1-alpha)*b[2],
          alpha*pb[3] + (1-alpha)*b[3],
        ];
        outB.push(sm);
        outS.push(Math.max(s, this._prevScores[bestJ] * 0.9)); // confiance “stable”
        usedPrev[bestJ] = 1;
      } else {
        // nouvelle apparition
        outB.push(b);
        outS.push(s);
      }
      if (outB.length >= limit) break;
    }

    // MàJ mémoire interne pour la prochaine frame
    this._prevBoxes = outB.slice(0, limit);
    this._prevScores = outS.slice(0, limit);

    return { boxes: outB, scores: outS };
  }

  /**
   * Méthode publique principale
   * Décode la sortie YOLO, applique le post-traitement et le lissage temporel
   */
  public process(
    results: Record<string, ortTypes.Tensor>,
    letter: Letterbox
  ): Detection[] {
    const outs = Object.values(results);
    const allBoxes:number[][] = [];
    const allScores:number[] = [];

    const minSidePx = Math.max(8, Math.min(letter.rw, letter.rh) * MIN_SIDE_FRAC);

    // Utilise les specs pré-calculées
    const sumHeads = this.headSpecs.reduce((a,b)=>a+b.count, 0);

    for (let tIdx = 0; tIdx < outs.length; tIdx++) {
      const acc = makeAccessor(outs[tIdx]); // Util globale
      if (!acc) continue;
      const { N, C, get } = acc;

      const numClasses = C - 5;
      if (numClasses < 1) continue;

      const headBoxes:number[][] = [];
      const headScores:number[] = [];

      const appearsFlattened = (N === sumHeads && sumHeads > 0);

      for (let i=0; i<N; i++) {
        let cx:number, cy:number, stride:number;

        if (appearsFlattened) {
          const gi = indexToGrid(i, this.headSpecs); // Util globale
          cx = gi.cx; cy = gi.cy; stride = gi.stride;
        } else {
          const g = Math.max(1, (Math.sqrt(N) + 0.5) | 0);
          stride = Math.max(8, Math.round(Math.max(letter.inw, letter.inh) / g));
          cx = i % g;
          cy = (i - cx) / g;
        }

        const tx = get(0,i),  ty = get(1,i),  tw = get(2,i),  th = get(3,i);
        const to = get(4,i);
        const tPerson = get(5, i); // class 0 = person

        const pObj = sigmoid(to); // Util globale
        if (pObj < OBJ_THRESH) continue;
        const pCls = sigmoid(tPerson); // Util globale
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

      // TOPK par sortie (réduit le bruit avant fusion)
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

    // 1) WBF un peu plus agressif
    const fused = weightedBoxesFusion(allBoxes, allScores, /* iouThr */ 0.45, /* limit */ MAX_DETECTIONS * 2); // Util globale

    // 2) Filtre centre: enlève les doublons au même centre (même personne, strides diff)
    const centerKeep = suppressByCenter(fused.boxes, fused.scores, /* centerFrac */ 0.40); // Util globale
    let cBoxes = centerKeep.map(i => fused.boxes[i]);
    let cScores = centerKeep.map(i => fused.scores[i]);

    // 3) Lissage & dédoublonnage temporels (APPEL INTERNE)
    const temp = this.temporalSmoothAndDedup(cBoxes, cScores, {
      iouGate: 0.35,
      centerFrac: 0.60,
      alpha: 0.65,
      limit: MAX_DETECTIONS
    });
    cBoxes = temp.boxes;
    cScores = temp.scores;

    // 4) NMS final serré (nettoyage ultime)
    const keep = nms(cBoxes, cScores, /* iou */ 0.25, /* limit */ MAX_DETECTIONS); // Util globale

    const out = new Array(keep.length);
    for (let i=0; i<keep.length; i++) {
      const k = keep[i];
      out[i] = { box: cBoxes[k], score: cScores[k], cls: 0, label: 'person' as const };
    }
    return out;
  }
}
// =======================================================
// === FIN DE LA CLASSE                                ===
// =======================================================


// ---------- Chargement ORT (WebGPU-first) ----------
async function loadOrtWebGPU(): Promise<void> {
  const hasWebGPU = typeof navigator !== 'undefined' && !!(navigator as any).gpu;
  if (!hasWebGPU) {
    throw new Error('WebGPU non disponible. Active-le et utilise HTTPS.');
  }

  const ortWebgpu = await import('onnxruntime-web/webgpu');

  try { (ortWebgpu as any).env.logLevel = 'warning'; } catch {}
  (ortWebgpu as any).env.wasm = (ortWebgpu as any).env.wasm || {};
  (ortWebgpu as any).env.wasm.wasmPaths =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

  (ortWebgpu as any).env.webgpu = (ortWebgpu as any).env.webgpu || {};
  (ortWebgpu as any).env.webgpu.powerPreference = 'high-performance';
  (ortWebgpu as any).env.webgpu.enableGraphCapture = true; // perfs max si forme fixe
  (ortWebgpu as any).env.webgpu.preferredOutputLocation = 'gpu-buffer';

  // @ts-ignore
  ort = ortWebgpu;
}

async function createSessionWebGPU() {
  session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['webgpu'] as any,
    graphOptimizationLevel: 'all',
    enableProfiling: false,
  });
  INPUT_NAME = session.inputNames[0];
  const meta: any = (session as any).inputMetadata?.[INPUT_NAME];
  const dims: number[] = meta?.dimensions ?? [];

  // Détecte layout + taille d'entrée
  if (Array.isArray(dims) && dims.length === 4) {
    if (dims[1] === 3) {
      INPUT_LAYOUT = 'NCHW';
      const H = dims[2], W = dims[3];
      if (Number.isFinite(H) && H! > 0 && Number.isFinite(W) && W! > 0) {
        FIXED_H = H!; FIXED_W = W!;
      }
    } else if (dims[3] === 3) {
      INPUT_LAYOUT = 'NHWC';
      const H = dims[1], W = dims[2];
      if (Number.isFinite(H) && H! > 0 && Number.isFinite(W) && W! > 0) {
        FIXED_H = H!; FIXED_W = W!;
      }
    } else {
      INPUT_LAYOUT = 'NCHW';
    }
  } else {
    INPUT_LAYOUT = 'NCHW'; // défaut Ultralytics
  }

  // 🔒 Fallback : si le modèle est “dynamique” (ou dims manquants), on force 640×640
  if (!FIXED_W || !FIXED_H) {
    FIXED_W = DEFAULT_INPUT_SIZE;
    FIXED_H = DEFAULT_INPUT_SIZE;
    console.warn('[YOLO] Dims dynamiques/indéterminées → fallback fixe', FIXED_W, 'x', FIXED_H);
  }

  console.log('[YOLO] EPs: webgpu');
  console.log('[YOLO] Input dims:', dims, 'Layout:', INPUT_LAYOUT, 'Fixed:', FIXED_W, FIXED_H);
}


let running = false;
let usingVFC = false;

// ---------- Prétraitement ----------
function letterboxAndToTensor(vw:number, vh:number) {
  // On est en capture de graphe → toujours la même taille
  netW = FIXED_W!;
  netH = FIXED_H!;

  ensurePreprocessSize(netW, netH);

  const scale = Math.min(netW / vw, netH / vh);
  const nw = (vw * scale + 0.5) | 0;
  const nh = (vh * scale + 0.5) | 0;
  const dx = ((netW - nw) / 2 + 0.5) | 0;
  const dy = ((netH - nh) / 2 + 0.5) | 0;

  pctx.setTransform(1, 0, 0, 1, 0, 0);
  pctx.clearRect(0, 0, netW, netH);
  pctx.fillStyle = '#000';
  pctx.fillRect(0, 0, netW, netH);
  pctx.drawImage(videoEl!, 0, 0, vw, vh, dx, dy, nw, nh);

  const img = pctx.getImageData(0, 0, netW, netH).data;
  const SIZE = netW * netH;

  if (INPUT_LAYOUT === 'NCHW') {
    for (let i=0, p=0; i<SIZE; i++, p+=4) {
      nchwBuf![i]          = img[p]   * (1/255);
      nchwBuf![i + SIZE]   = img[p+1] * (1/255);
      nchwBuf![i + 2*SIZE] = img[p+2] * (1/255);
    }
  } else {
    for (let i=0, p=0, o=0; i<SIZE; i++, p+=4, o+=3) {
      nhwcBuf![o]   = img[p]   * (1/255);
      nhwcBuf![o+1] = img[p+1] * (1/255);
      nhwcBuf![o+2] = img[p+2] * (1/255);
    }
  }

  return {
    tensor: new ort.Tensor(
      'float32',
      INPUT_LAYOUT === 'NCHW' ? nchwBuf! : nhwcBuf!,
      INPUT_LAYOUT === 'NCHW' ? [1,3,netH,netW] : [1,netH,netW,3]
    ),
    letter: { sx: scale, sy: scale, dw: dx, dh: dy, rw: vw, rh: vh, inw: netW, inh: netH }
  };
}


// ---------- HUD amorti ----------
let hudPrepared = false;
let hudBoxW = 0;

function prepareHUDOnce(ctx: CanvasRenderingContext2D) {
  if (hudPrepared) return;
  ctx.save();
  ctx.font = '16px monospace';
  const w1 = ctx.measureText('Unique: 0000').width;
  const w2 = ctx.measureText('Active: 0000').width;
  hudBoxW = Math.max(w1, w2) + 16;
  ctx.restore();
  hudPrepared = true;
}

// ---------- Dessin avec tracks ----------
function drawTrackedDetections(
  tracks: Array<{ id: number; box: number[]; personId: number | null; framesSeen: number }>,
  uniqueCount: number,
  activeCount: number
) {
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
  overlayCtx!.font = '14px monospace';
  overlayCtx!.textBaseline = 'top';

  for (let i = 0; i < tracks.length; i++) {
    const track = tracks[i];
    const b = track.box;
    const x1 = b[0], y1 = b[1], x2 = b[2], y2 = b[3];

    const vx1 = MIRROR ? (vw - x2) : x1;
    const vx2 = MIRROR ? (vw - x1) : x2;

    const rx1 = ox + vx1 * scale;
    const ry1 = oy + y1 * scale;
    const rwc = (vx2 - vx1) * scale;
    const rhc = (y2 - y1) * scale;

    const isRecognized = track.personId !== null;
    const color = isRecognized ? '#00ff88' : '#ffaa00';

    overlayCtx!.strokeStyle = color;
    overlayCtx!.fillStyle = 'rgba(0,0,0,0.6)';
    overlayCtx!.strokeRect(rx1, ry1, rwc, rhc);

    const label = track.personId !== null ? `Person #${track.personId}` : `Track #${track.id}`;
    const tw = overlayCtx!.measureText(label).width + 8;
    overlayCtx!.fillRect(rx1, Math.max(0, ry1 - 20), tw, 18);
    overlayCtx!.fillStyle = color;
    overlayCtx!.fillText(label, rx1 + 4, ry1 - 18);
  }

  // HUD
  prepareHUDOnce(overlayCtx!);
  overlayCtx!.font = '16px monospace';
  overlayCtx!.fillStyle = 'rgba(0,0,0,0.7)';
  overlayCtx!.fillRect(10, 10, hudBoxW, 2*22 + 8);
  overlayCtx!.fillStyle = '#fbff68';
  overlayCtx!.fillText(`Unique: ${uniqueCount}`, 18, 16);
  overlayCtx!.fillText(`Active: ${activeCount}`, 18, 38);

  if (channelEl) {
    channelEl.textContent = uniqueCount > 0 ? `AV1 (${uniqueCount})` : 'AV1';
  }
}

// ---------- Boucle d'inférence ----------
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
  // <<< OPTIMISATION: Vérifie aussi yoloProcessor
  if (!session || !tracker || !yoloProcessor || videoEl!.readyState < 2) return;

  resizeOverlayToVideoIfNeeded();

  const vw = Math.max(1, videoEl!.videoWidth), vh = Math.max(1, videoEl!.videoHeight);
  const { tensor, letter } = letterboxAndToTensor(vw, vh);

  const feeds: Record<string, ortTypes.Tensor> = { [INPUT_NAME]: tensor };

  let results: Record<string, ortTypes.Tensor>;
  try {
    results = await session.run(feeds);
  } catch (e: any) {
    const msg = String(e?.message || e);

    // Softmax mal placé (rare avec bons exports)
    const softmaxBadAxis =
      msg.includes('softmax only supports last axis for now') ||
      (msg.includes('Softmax') && msg.toLowerCase().includes('failed'));
    if (softmaxBadAxis) {
      throw new Error(
        'Le modèle utilise Softmax sur un axe non supporté en WebGPU. ' +
        'Patch l’ONNX puis réessaie.'
      );
    }

    // Dimensions invalides => typiquement modèle fixe et entrée non 640x640
    if (msg.includes('Got invalid dimensions for input')) {
      throw new Error(
        'Dimensions d’entrée invalides. Assure-toi que netW/netH correspondent à la taille fixe du modèle (ex: 640×640).'
      );
    }

    // On relance l’erreur telle quelle pour ne pas masquer les vrais soucis
    throw e;
  }

  // <<< OPTIMISATION: Utilise l'instance de la classe
  const detections = yoloProcessor.process(results, letter);
  
  const { tracks, uniqueCount, activeCount } = await tracker.update(detections);
  drawTrackedDetections(tracks, uniqueCount, activeCount);
}

async function loop() {
  if (!running) return;
  await inferOnce();
  if (running) scheduleNext(loop);
}

// ---------- Init / API publique ----------
async function initDetector() {
  await loadOrtWebGPU();
  await createSessionWebGPU();

  // Initialiser le tracker ReID
  tracker = new PersonTracker(videoEl!);
  await tracker.initialize(ort);

  // <<< OPTIMISATION: Initialise le processeur YOLO
  yoloProcessor = new YoloV8Processor(FIXED_W!, FIXED_H!);

  // Warmup YOLO: tensor exactement à la taille d’entrée du modèle
  const warmW = FIXED_W!;
  const warmH = FIXED_H!;
  ensurePreprocessSize(warmW, warmH);
  netW = warmW; netH = warmH;

  const zero = new Float32Array(3 * warmW * warmH);
  const warm = new ort.Tensor(
    'float32',
    zero,
    INPUT_LAYOUT === 'NCHW'
      ? [1,3,warmH,warmW]
      : [1,warmH,warmW,3]
  );
  const feeds: Record<string, ortTypes.Tensor> = { [INPUT_NAME]: warm };
  try { await session!.run(feeds); } catch { /* silencieux */ }

  console.log('[Init] Détecteur et tracker prêts !');
}

export async function startDetection() {
  // <<< OPTIMISATION: Vérifie aussi yoloProcessor
  if (!session || !tracker || !yoloProcessor) await initDetector();

  usingVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

  running = true;
  scheduleNext(loop);
}

export function stopDetection() {
  running = false;
  cancelScheduled();
  overlayCtx!.clearRect(0, 0, overlay!.width, overlay!.height);
  yoloProcessor?.reset(); // <<< AJOUT: Réinitialise l'état du lissage
}

// Expose pour index.html
;(window as any).startDetection = startDetection;
;(window as any).stopDetection = stopDetection;