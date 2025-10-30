// tracker.ts — Système de tracking ULTRA ROBUSTE avec ReID + signature couleur + contexte spatial

import type * as ortTypes from 'onnxruntime-web';

// ==================== CONFIG ROBUSTE ====================
const REID_MODEL_PATH = '/osnet_x0_25_msmt17_webgpu_noL2.onnx';
const REID_INPUT_SIZE = 128;
const REID_INPUT_HEIGHT = 256;

// === STRATÉGIE: TRÈS CONSERVATRICE (éviter les fusions à tout prix) ===
const STRICTNESS = 0.85; // Maximum de prudence

// Seuils ReID ultra stricts
const BASE_SIM_MATCH_STRICT = 0.85; // Très haut: seules les correspondances ÉVIDENTES
const BASE_SIM_NOVELTY      = 0.72; // Tout ce qui est en-dessous = nouvelle personne

const SIM_MATCH_STRICT = BASE_SIM_MATCH_STRICT + 0.08 * STRICTNESS; // ~0.926
const SIM_NOVELTY      = BASE_SIM_NOVELTY      + 0.06 * STRICTNESS; // ~0.777

const AMBIG_MARGIN = 0.008; // Marge minimale

// === PONDÉRATION: Embedding est ROI ===
const EMB_WEIGHT = 0.85; // L'embedding décide en priorité
const IOU_WEIGHT = 0.15;

// === SPATIAL: Association prudente ===
const IOU_THRESHOLD = 0.25; // Plus permissif pour tracker la même personne qui bouge

// === TIMEOUTS ===
const TRACK_TIMEOUT_MS = 1200;        // Disparition rapide pour éviter confusion
const PERSON_MEMORY_MS = 120000;      // 2min seulement (focus sur le présent)
const NEW_PERSON_COOLDOWN_MS = 1200;  // Long cooldown anti-doublons

// === MÉMOIRE EMBEDDINGS ===
const MAX_EMB_PER_PERSON = 8; // Moins d'historique = moins de drift
const MIN_EMB_FOR_CENTROID = 3; // Besoin de plusieurs observations avant moyenne

// === SIGNATURE COULEUR HAUTE RÉSOLUTION ===
const COLOR_H_BINS = 16; // Maximum de précision
const COLOR_S_BINS = 5;
const COLOR_V_BINS = 5;
const COLOR_SIG_LEN = COLOR_H_BINS * COLOR_S_BINS * COLOR_V_BINS;

const COLOR_SIM_MATCH_MIN = 0.72; // Très strict sur couleur
const COLOR_STRICT_BOOST  = 0.12; // Gros penalty si couleur différente
const COLOR_WEIGHT_IN_DECISION = 0.35; // Poids de la couleur dans décision finale

// === VALIDATION MULTI-CRITÈRES ===
const MIN_FRAMES_FOR_STABLE_ID = 5; // Observations minimales avant ID stable
const BBOX_SIZE_SIMILARITY_THRESHOLD = 0.6; // Tailles de boîtes doivent être cohérentes

// ==================== TYPES ====================
export interface Detection {
  box: number[];
  score: number;
  cls: number;
  label: string;
}

interface Track {
  id: number;
  box: number[];
  embedding: Float32Array;
  lastSeen: number;
  framesSeen: number;
  personId: number | null;
  colorSig: Float32Array;
  confirmedId: boolean; // ID confirmé après plusieurs frames
}

interface Person {
  id: number;
  embeddings: Float32Array[];
  centroid: Float32Array | null;
  colorSigAvg: Float32Array | null;
  firstSeen: number;
  lastSeen: number;
  timesReentered: number;
  lastNewNearTs: number;
  avgBoxSize: number; // Taille moyenne des boîtes
  observationCount: number; // Nombre total d'observations
}

// ==================== UTILS ====================
function iou(a: number[], b: number[]): number {
  const [ax1, ay1, ax2, ay2] = a;
  const [bx1, by1, bx2, by2] = b;
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = (ax2 - ax1) * (ay2 - ay1);
  const areaB = (bx2 - bx1) * (by2 - by1);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

function boxSize(box: number[]): number {
  return (box[2] - box[0]) * (box[3] - box[1]);
}

function boxSizeSimilarity(box1: number[], box2: number[]): number {
  const s1 = boxSize(box1);
  const s2 = boxSize(box2);
  const minS = Math.min(s1, s2);
  const maxS = Math.max(s1, s2);
  return maxS > 0 ? minS / maxS : 0;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  const L = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < L; i++) {
    const ai = a[i], bi = b[i];
    dot += ai * bi; na += ai * ai; nb += bi * bi;
  }
  const denom = Math.sqrt(Math.max(1e-12, na)) * Math.sqrt(Math.max(1e-12, nb));
  return denom > 0 ? dot / denom : 0;
}

function robustAverageEmbedding(embeddings: Float32Array[]): Float32Array {
  if (embeddings.length === 0) return new Float32Array(0);
  if (embeddings.length === 1) return embeddings[0].slice();
  
  // Utiliser les N derniers embeddings seulement
  const recent = embeddings.slice(-MAX_EMB_PER_PERSON);
  const d = recent[0].length;
  const avg = new Float32Array(d);
  
  for (let i = 0; i < recent.length; i++) {
    const e = recent[i];
    const L = Math.min(d, e.length);
    for (let j = 0; j < L; j++) avg[j] += e[j];
  }
  
  for (let j = 0; j < d; j++) avg[j] /= recent.length;
  
  // Normalisation L2
  let norm = 0;
  for (let j = 0; j < d; j++) norm += avg[j] * avg[j];
  norm = Math.sqrt(norm) + 1e-6;
  for (let j = 0; j < d; j++) avg[j] /= norm;
  
  return avg;
}

function emaUpdateCentroid(prev: Float32Array | null, emb: Float32Array, alpha = 0.10): Float32Array {
  if (!prev || prev.length !== emb.length) return emb.slice();
  const out = prev.slice();
  for (let i = 0; i < out.length; i++) out[i] = alpha * emb[i] + (1 - alpha) * out[i];
  
  // Re-normaliser
  let norm = 0;
  for (let i = 0; i < out.length; i++) norm += out[i] * out[i];
  norm = Math.sqrt(norm) + 1e-6;
  for (let i = 0; i < out.length; i++) out[i] /= norm;
  
  return out;
}

function emaUpdateColor(prev: Float32Array | null, sig: Float32Array, alpha = 0.15): Float32Array {
  if (!prev || prev.length !== sig.length) return sig.slice();
  const out = prev.slice();
  for (let i = 0; i < out.length; i++) out[i] = alpha * sig[i] + (1 - alpha) * out[i];
  return out;
}

// ==================== COULEUR HSV ====================
function rgb2hsv(r:number,g:number,b:number){
  const mx=Math.max(r,g,b), mn=Math.min(r,g,b), d=mx-mn;
  let h=0, s=mx===0?0:d/mx, v=mx;
  if (d!==0){
    if (mx===r) h=(g-b)/d + (g<b?6:0);
    else if (mx===g) h=(b-r)/d + 2;
    else h=(r-g)/d + 4;
    h/=6;
  }
  return [h,s,v] as const;
}

function computeColorSigFromPixels(pixels:Uint8ClampedArray, W:number, H:number): Float32Array {
  const HB=COLOR_H_BINS, SB=COLOR_S_BINS, VB=COLOR_V_BINS;
  const bins = new Float32Array(HB*SB*VB);
  
  // Zone torse centrale (évite fond, tête, jambes)
  const x0=Math.floor(W*0.20), x1=Math.ceil(W*0.80);
  const y0=Math.floor(H*0.30), y1=Math.ceil(H*0.75);
  
  let count = 0;
  for (let y=y0; y<y1; y++) {
    for (let x=x0; x<x1; x++) {
      const p=(y*W + x)*4;
      const r=pixels[p]/255, g=pixels[p+1]/255, b=pixels[p+2]/255;
      
      // Filtrer pixels trop sombres/trop clairs (évite ombres/reflets)
      const [h,s,v]=rgb2hsv(r,g,b);
      if (v < 0.15 || v > 0.95) continue; // Skip extrêmes
      
      const hi=Math.min(HB-1, Math.floor(h*HB));
      const si=Math.min(SB-1, Math.floor(s*SB));
      const vi=Math.min(VB-1, Math.floor(v*VB));
      bins[hi*SB*VB + si*VB + vi] += 1;
      count++;
    }
  }
  
  // L1-normalisation
  if (count > 0) for (let i=0;i<bins.length;i++) bins[i]/=count;
  return bins;
}

function cosineFloat(a: Float32Array, b: Float32Array): number {
  const L=Math.min(a.length,b.length);
  let dot=0, na=0, nb=0;
  for(let i=0;i<L;i++){ const x=a[i], y=b[i]; dot+=x*y; na+=x*x; nb+=y*y; }
  const denom=Math.sqrt(Math.max(1e-12,na))*Math.sqrt(Math.max(1e-12,nb));
  return denom>0? dot/denom : 0;
}

// ==================== TRACKER CLASS ====================
export class PersonTracker {
  private ort: typeof import('onnxruntime-web') | null = null;
  private reidSession: ortTypes.InferenceSession | null = null;

  private reidBatch = 1;
  private reidLayout: 'NCHW' | 'NHWC' = 'NCHW';
  private reidW = REID_INPUT_SIZE;
  private reidH = REID_INPUT_HEIGHT;
  private inputName = '';
  private outputName = '';

  private activeTracks: Map<number, Track> = new Map();
  private nextTrackId = 1;

  private persons: Map<number, Person> = new Map();
  private nextPersonId = 1;

  private cropCanvas: HTMLCanvasElement;
  private cropCtx: CanvasRenderingContext2D;
  private videoElement: HTMLVideoElement;

  constructor(videoElement: HTMLVideoElement) {
    this.videoElement = videoElement;
    this.cropCanvas = document.createElement('canvas');
    this.cropCanvas.width = REID_INPUT_SIZE;
    this.cropCanvas.height = REID_INPUT_HEIGHT;
    this.cropCtx = this.cropCanvas.getContext('2d', { willReadFrequently: true })!;
  }

  // ==================== INIT ====================
  async initialize(ortInstance: typeof import('onnxruntime-web')) {
    this.ort = ortInstance;
    try { if ((this.ort as any).env) (this.ort as any).env.logLevel = 'warning'; } catch {}

    console.log('[Tracker] Chargement du modèle ReID…', REID_MODEL_PATH);
    const res = await fetch(REID_MODEL_PATH, { cache: 'no-store' });
    if (!res.ok) throw new Error(`[ReID] HTTP ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());

    try {
      this.reidSession = await this.ort!.InferenceSession.create(buf, {
        executionProviders: ['webgpu'] as any,
        graphOptimizationLevel: 'all',
      });
      console.log('[Tracker] ReID WebGPU ✅ (opt=all)');
    } catch (e1) {
      console.warn('[Tracker] Fallback opt=disabled', e1);
      this.reidSession = await this.ort!.InferenceSession.create(buf, {
        executionProviders: ['webgpu'] as any,
        graphOptimizationLevel: 'disabled',
      });
      console.log('[Tracker] ReID WebGPU ✅ (opt=disabled)');
    }

    this.inputName = this.reidSession.inputNames[0];
    this.outputName = this.reidSession.outputNames[0];

    const metaIn: any = (this.reidSession as any).inputMetadata?.[this.inputName];
    const dims: number[] = metaIn?.dimensions ?? [];
    console.log('[ReID] input dims:', dims);

    if (dims.length === 4) {
      this.reidBatch = typeof dims[0] === 'number' && dims[0] > 0 ? dims[0] : 1;
      if (dims[1] === 3) {
        this.reidLayout = 'NCHW';
        this.reidH = (typeof dims[2] === 'number' && dims[2] > 0) ? dims[2] : REID_INPUT_HEIGHT;
        this.reidW = (typeof dims[3] === 'number' && dims[3] > 0) ? dims[3] : REID_INPUT_SIZE;
      } else if (dims[3] === 3) {
        this.reidLayout = 'NHWC';
        this.reidH = (typeof dims[1] === 'number' && dims[1] > 0) ? dims[1] : REID_INPUT_HEIGHT;
        this.reidW = (typeof dims[2] === 'number' && dims[2] > 0) ? dims[2] : REID_INPUT_SIZE;
      }
    }

    this.cropCanvas.width = this.reidW;
    this.cropCanvas.height = this.reidH;

    console.log('[ReID] Configuré:', this.reidBatch, this.reidLayout, this.reidW, 'x', this.reidH);
  }

  // ==================== EMBEDDING + COULEUR ====================
  private async extractEmbeddingAndColor(box: number[]): Promise<{emb: Float32Array | null, color: Float32Array | null}> {
    if (!this.reidSession) return { emb: null, color: null };

    const [x1, y1, x2, y2] = box;
    const vw = this.videoElement.videoWidth;
    const vh = this.videoElement.videoHeight;

    const cx1 = Math.max(0, Math.min(vw, x1));
    const cy1 = Math.max(0, Math.min(vh, y1));
    const cx2 = Math.max(0, Math.min(vw, x2));
    const cy2 = Math.max(0, Math.min(vh, y2));

    const w = cx2 - cx1;
    const h = cy2 - cy1;

    if (w < 15 || h < 30) return { emb: null, color: null }; // Filtrage strict

    this.cropCtx.clearRect(0, 0, this.reidW, this.reidH);
    this.cropCtx.drawImage(
      this.videoElement,
      cx1, cy1, w, h,
      0, 0, this.reidW, this.reidH
    );

    const imgData = this.cropCtx.getImageData(0, 0, this.reidW, this.reidH);
    const pixels = imgData.data;

    const colorSig = computeColorSigFromPixels(pixels, this.reidW, this.reidH);

    const SIZE = this.reidW * this.reidH;
    const C = 3;
    const one = new Float32Array(C * SIZE);
    const mean = [0.485, 0.456, 0.406];
    const std  = [0.229, 0.224, 0.225];

    if (this.reidLayout === 'NCHW') {
      for (let i = 0, p = 0; i < SIZE; i++, p += 4) {
        const r = pixels[p] / 255, g = pixels[p + 1] / 255, b = pixels[p + 2] / 255;
        one[i]           = (r - mean[0]) / std[0];
        one[i +   SIZE]  = (g - mean[1]) / std[1];
        one[i + 2*SIZE]  = (b - mean[2]) / std[2];
      }
    } else {
      for (let i = 0, p = 0, o = 0; i < SIZE; i++, p += 4, o += 3) {
        const r = pixels[p] / 255, g = pixels[p + 1] / 255, b = pixels[p + 2] / 255;
        one[o]   = (r - mean[0]) / std[0];
        one[o+1] = (g - mean[1]) / std[1];
        one[o+2] = (b - mean[2]) / std[2];
      }
    }

    const B = this.reidBatch > 0 ? this.reidBatch : 1;
    const perSample = one.length;
    let data: Float32Array;

    if (B === 1) {
      data = one;
    } else {
      data = new Float32Array(B * perSample);
      data.set(one, 0);
    }

    const shape =
      this.reidLayout === 'NCHW'
        ? [B, 3, this.reidH, this.reidW]
        : [B, this.reidH, this.reidW, 3];

    const inputTensor = new this.ort!.Tensor('float32', data, shape);

    try {
      const results = await this.reidSession.run({ [this.inputName]: inputTensor });
      const outAny = results[this.outputName];
      const raw = outAny.data as Float32Array;

      let emb: Float32Array;
      if (outAny.dims && outAny.dims.length === 2 && outAny.dims[0] >= 1) {
        const d = outAny.dims[1];
        emb = raw.subarray(0, d).slice();
      } else {
        emb = raw.slice();
      }

      // Normalisation L2
      let norm = 0;
      for (let i = 0; i < emb.length; i++) norm += emb[i] * emb[i];
      norm = Math.sqrt(norm) + 1e-6;
      for (let i = 0; i < emb.length; i++) emb[i] /= norm;

      return { emb, color: colorSig };
    } catch (e) {
      console.error('[Tracker] Erreur extraction:', e);
      return { emb: null, color: null };
    }
  }

  // ==================== MATCHING ROBUSTE ====================
  private matchDetectionToTrack(
    detection: Detection,
    embedding: Float32Array,
    colorSig: Float32Array | null
  ): Track | null {
    let bestTrack: Track | null = null;
    let bestScore = -1;

    for (const track of this.activeTracks.values()) {
      const iouScore = iou(detection.box, track.box);
      if (iouScore < IOU_THRESHOLD) continue; // Pré-filtrage spatial
      
      const embSim = cosineSimilarity(embedding, track.embedding);
      
      // Vérification taille de boîte
      const sizeSim = boxSizeSimilarity(detection.box, track.box);
      if (sizeSim < BBOX_SIZE_SIMILARITY_THRESHOLD) continue; // Tailles trop différentes
      
      // Vérification couleur
      let colorPenalty = 0;
      if (colorSig && track.colorSig.length > 0) {
        const colorSim = cosineFloat(colorSig, track.colorSig);
        if (colorSim < COLOR_SIM_MATCH_MIN) colorPenalty = 0.15; // Pénalité importante
      }
      
      const combinedScore = (IOU_WEIGHT * iouScore + EMB_WEIGHT * embSim) * (1 - colorPenalty);

      if (combinedScore > bestScore) {
        bestScore = combinedScore;
        bestTrack = track;
      }
    }

    return bestTrack;
  }

  private bestPersonForEmbedding(
    embedding: Float32Array,
    colorSig: Float32Array | null,
    box: number[]
  ): { person: Person | null; sim: number; colorSim: number } {
    let best: Person | null = null;
    let bestSim = -1;
    let bestColorSim = 0;
    const now = Date.now();

    for (const person of this.persons.values()) {
      if (now - person.lastSeen > PERSON_MEMORY_MS) continue;
      
      // Vérifier taille de boîte cohérente
      if (person.avgBoxSize > 0) {
        const currentSize = boxSize(box);
        const sizeRatio = Math.min(currentSize, person.avgBoxSize) / Math.max(currentSize, person.avgBoxSize);
        if (sizeRatio < BBOX_SIZE_SIMILARITY_THRESHOLD) continue;
      }
      
      // Embedding
      const ref = person.centroid ?? robustAverageEmbedding(person.embeddings);
      if (!ref.length) continue;
      const embSim = cosineSimilarity(embedding, ref);
      
      // Couleur
      let colorSim = 1.0;
      if (colorSig && person.colorSigAvg) {
        colorSim = cosineFloat(colorSig, person.colorSigAvg);
      }
      
      // Score combiné (embedding + couleur)
      const combinedSim = embSim * (1 - COLOR_WEIGHT_IN_DECISION) + colorSim * COLOR_WEIGHT_IN_DECISION;
      
      if (combinedSim > bestSim) {
        bestSim = combinedSim;
        best = person;
        bestColorSim = colorSim;
      }
    }
    
    return { person: best, sim: bestSim, colorSim: bestColorSim };
  }

  private attachEmbeddingToPerson(person: Person, emb: Float32Array, now: number, box: number[], color?: Float32Array | null) {
    person.embeddings.push(emb);
    if (person.embeddings.length > MAX_EMB_PER_PERSON) person.embeddings.shift();
    
    // Centroid seulement après plusieurs observations
    if (person.embeddings.length >= MIN_EMB_FOR_CENTROID) {
      person.centroid = emaUpdateCentroid(person.centroid, emb, 0.10);
    }
    
    if (color) person.colorSigAvg = emaUpdateColor(person.colorSigAvg ?? null, color, 0.15);
    
    // Mise à jour taille moyenne
    const currentSize = boxSize(box);
    if (person.avgBoxSize === 0) {
      person.avgBoxSize = currentSize;
    } else {
      person.avgBoxSize = 0.8 * person.avgBoxSize + 0.2 * currentSize;
    }
    
    person.lastSeen = now;
    person.observationCount++;
  }

  private createNewPerson(emb: Float32Array, now: number, box: number[], color?: Float32Array | null): Person {
    const p: Person = {
      id: this.nextPersonId++,
      embeddings: [emb],
      centroid: null, // Pas de centroid immédiat
      colorSigAvg: color ? color.slice() : new Float32Array(COLOR_SIG_LEN),
      firstSeen: now,
      lastSeen: now,
      timesReentered: 0,
      lastNewNearTs: now,
      avgBoxSize: boxSize(box),
      observationCount: 1,
    };
    this.persons.set(p.id, p);
    console.log(`[Tracker] 🆕 Nouvelle personne #${p.id}`);
    return p;
  }

  // ==================== UPDATE ====================
  async update(detections: Detection[]): Promise<{
    tracks: Track[];
    uniqueCount: number;
    activeCount: number;
  }> {
    const now = Date.now();

    // 1) Extraire features
    const dets: Array<{ det: Detection; emb: Float32Array | null; color: Float32Array | null }> = [];
    for (const det of detections) {
      const both = await this.extractEmbeddingAndColor(det.box);
      dets.push({ det, emb: both.emb, color: both.color });
    }

    // 2) Associer aux tracks actifs (continuité temporelle)
    const matchedTracks = new Set<number>();
    const matchedDetections = new Set<number>();

    for (let i = 0; i < dets.length; i++) {
      const { det, emb, color } = dets[i];
      if (!emb) continue;

      const matchedTrack = this.matchDetectionToTrack(det, emb, color);
      if (matchedTrack) {
        matchedTrack.box = det.box;
        matchedTrack.embedding = emb;
        if (color) matchedTrack.colorSig = color;
        matchedTrack.lastSeen = now;
        matchedTrack.framesSeen++;
        
        // Confirmation ID après plusieurs frames
        if (matchedTrack.framesSeen >= MIN_FRAMES_FOR_STABLE_ID) {
          matchedTrack.confirmedId = true;
        }

        matchedTracks.add(matchedTrack.id);
        matchedDetections.add(i);
        
        // Mettre à jour la personne associée
        if (matchedTrack.personId !== null) {
          const person = this.persons.get(matchedTrack.personId);
          if (person) {
            this.attachEmbeddingToPerson(person, emb, now, det.box, color || undefined);
          }
        }
      }
    }

    // 3) Nouvelles détections: créer tracks + assigner identités
    for (let i = 0; i < dets.length; i++) {
      if (matchedDetections.has(i)) continue;

      const { det, emb, color } = dets[i];
      if (!emb) continue;

      const { person: best, sim: bestSim, colorSim } = this.bestPersonForEmbedding(emb, color, det.box);

      let decidedPerson: Person | null = null;

      if (best) {
        // Vérification couleur stricte
        const colorOk = colorSim >= COLOR_SIM_MATCH_MIN;
        const dynMatchStrict = colorOk ? SIM_MATCH_STRICT : (SIM_MATCH_STRICT + COLOR_STRICT_BOOST);

        if (bestSim >= dynMatchStrict && colorOk) {
          // Correspondance TRÈS forte
          decidedPerson = best;
          console.log(`[Tracker] ✅ Réassociation personne #${best.id} (sim=${bestSim.toFixed(3)}, color=${colorSim.toFixed(3)})`);
        } else if (bestSim <= SIM_NOVELTY - AMBIG_MARGIN || !colorOk) {
          // Clairement nouvelle personne
          
          // Anti-doublons: vérifier qu'aucune personne similaire n'a été créée récemment
          let tooSoon = false;
          for (const p of this.persons.values()) {
            if (now - p.lastNewNearTs > NEW_PERSON_COOLDOWN_MS) continue;
            
            if (color && p.colorSigAvg) {
              const cSim = cosineFloat(color, p.colorSigAvg);
              if (cSim >= COLOR_SIM_MATCH_MIN * 0.9) { // Légèrement plus permissif
                const pEmb = p.centroid ?? robustAverageEmbedding(p.embeddings);
                if (pEmb.length > 0) {
                  const eSim = cosineSimilarity(emb, pEmb);
                  if (eSim >= SIM_NOVELTY * 0.85) { // Dans la zone grise
                    tooSoon = true;
                    decidedPerson = p; // Rattacher à la personne existante
                    break;
                  }
                }
              }
            }
          }
          
          if (!tooSoon) {
            decidedPerson = this.createNewPerson(emb, now, det.box, color);
          }
        } else {
          // Zone d'ambiguïté: utiliser contexte spatial
          let chosen: Person | null = null;
          let maxIou = 0;
          
          for (const t of this.activeTracks.values()) {
            if (t.personId == null || !t.confirmedId) continue; // Seulement IDs confirmés
            const iouv = iou(det.box, t.box);
            if (iouv > maxIou) {
              maxIou = iouv;
              const candidate = this.persons.get(t.personId);
              if (candidate) chosen = candidate;
            }
          }
          
          // Seuil spatial strict
          if (chosen && maxIou >= 0.40) {
            decidedPerson = chosen;
            console.log(`[Tracker] 🔄 Association spatiale personne #${chosen.id} (iou=${maxIou.toFixed(3)})`);
          } else {
            // En cas de doute: créer nouvelle personne (principe de prudence)
            decidedPerson = this.createNewPerson(emb, now, det.box, color);
          }
        }
      } else {
        // Aucune personne existante: nouvelle identité
        decidedPerson = this.createNewPerson(emb, now, det.box, color);
      }

      // Créer un nouveau track
      const newTrack: Track = {
        id: this.nextTrackId++,
        box: det.box,
        embedding: emb,
        lastSeen: now,
        framesSeen: 1,
        personId: decidedPerson ? decidedPerson.id : null,
        colorSig: color ?? new Float32Array(0),
        confirmedId: false,
      };
      this.activeTracks.set(newTrack.id, newTrack);
      
      // Attacher à la personne
      if (decidedPerson) {
        this.attachEmbeddingToPerson(decidedPerson, emb, now, det.box, color || undefined);
      }
    }

    // 4) Timeout des tracks inactifs
    for (const [trackId, track] of this.activeTracks.entries()) {
      if (now - track.lastSeen > TRACK_TIMEOUT_MS) {
        const p = track.personId ? this.persons.get(track.personId) : null;
        if (p) p.timesReentered++;
        this.activeTracks.delete(trackId);
      }
    }

    // 5) Garbage collection des personnes anciennes
    for (const [personId, person] of this.persons.entries()) {
      if (now - person.lastSeen > PERSON_MEMORY_MS) {
        console.log(`[Tracker] 🗑️ Suppression personne #${personId} (timeout)`);
        this.persons.delete(personId);
      }
    }

    return {
      tracks: Array.from(this.activeTracks.values()),
      uniqueCount: this.persons.size,
      activeCount: this.activeTracks.size
    };
  }

  // ==================== GETTERS ====================
  getUniqueCount(): number {
    return this.persons.size;
  }

  getActiveCount(): number {
    return this.activeTracks.size;
  }

  getPersons(): Person[] {
    return Array.from(this.persons.values());
  }

  reset() {
    this.activeTracks.clear();
    this.persons.clear();
    this.nextTrackId = 1;
    this.nextPersonId = 1;
    console.log('[Tracker] 🔄 Reset complet');
  }
}