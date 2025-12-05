import { MODEL_WIDTH, MODEL_HEIGHT } from './config.js';

export class DetectionEngine {
  constructor() {
    this.session = null;
    this.inputName = null;
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = MODEL_WIDTH;
    this.offscreenCanvas.height = MODEL_HEIGHT;
    this.offscreenCtx = this.offscreenCanvas.getContext('2d', { willReadFrequently: true });
    this.confThreshold = 0.6;
  }

  async initialize() {
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU non disponible (Chrome/Edge récent nécessaire).');
    }

    try {
      this.session = await ort.InferenceSession.create('model/yolov8n.onnx', {
        executionProviders: ['webgpu']
      });
      this.inputName = this.session.inputNames[0];
      console.log('Session YOLO initialisée');
    } catch (e) {
      console.error('Erreur init YOLO:', e);
      throw e;
    }
  }

  setConfidenceThreshold(value) {
    this.confThreshold = value;
  }

  preprocess(videoElement) {
    this.offscreenCtx.drawImage(videoElement, 0, 0, MODEL_WIDTH, MODEL_HEIGHT);
    const imageData = this.offscreenCtx.getImageData(0, 0, MODEL_WIDTH, MODEL_HEIGHT);
    const { data } = imageData;
    const float32Data = new Float32Array(3 * MODEL_WIDTH * MODEL_HEIGHT);
    const size = MODEL_WIDTH * MODEL_HEIGHT;

    for (let i = 0; i < size; i++) {
      const r = data[4*i] / 255;
      const g = data[4*i+1] / 255;
      const b = data[4*i+2] / 255;
      float32Data[i] = r;
      float32Data[i+size] = g;
      float32Data[i+2*size] = b;
    }

    return new ort.Tensor('float32', float32Data, [1, 3, MODEL_HEIGHT, MODEL_WIDTH]);
  }

  postprocess(output, canvasWidth, canvasHeight) {
    const data = output.data;
    const dims = output.dims;
    const channels = dims[1];
    const numAnchors = dims[2];
    const numClasses = channels - 4;
    const boxes = [];

    for (let i = 0; i < numAnchors; i++) {
      const cx = data[0 * numAnchors + i];
      const cy = data[1 * numAnchors + i];
      const w = data[2 * numAnchors + i];
      const h = data[3 * numAnchors + i];

      if (w <= 0 || h <= 0) continue;

      let bestScore = -Infinity, bestClass = -1;
      for (let c = 0; c < numClasses; c++) {
        const score = data[(4 + c) * numAnchors + i];
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      const prob = bestScore;
      if (bestClass !== 0 || prob < this.confThreshold) continue;

      let x1 = (cx - w/2) / MODEL_WIDTH;
      let y1 = (cy - h/2) / MODEL_HEIGHT;
      let x2 = (cx + w/2) / MODEL_WIDTH;
      let y2 = (cy + h/2) / MODEL_HEIGHT;

      if (x2 <= x1 || y2 <= y1 || x2 < 0 || y2 < 0 || x1 > 1 || y1 > 1) continue;

      x1 = Math.max(0, Math.min(1, x1));
      y1 = Math.max(0, Math.min(1, y1));
      x2 = Math.max(0, Math.min(1, x2));
      y2 = Math.max(0, Math.min(1, y2));

      boxes.push({
        x1, y1, x2, y2,
        score: prob,
        classId: bestClass
      });
    }

    const nmsBoxes = this.nonMaxSuppression(boxes);
    return nmsBoxes.map(b => ({
      x: b.x1 * canvasWidth,
      y: b.y1 * canvasHeight,
      w: (b.x2 - b.x1) * canvasWidth,
      h: (b.y2 - b.y1) * canvasHeight,
      score: b.score
    }));
  }

  nonMaxSuppression(boxes, iouThreshold = 0.45) {
    const sorted = boxes.slice().sort((a, b) => b.score - a.score);
    const result = [];

    while (sorted.length > 0) {
      const candidate = sorted.shift();
      result.push(candidate);

      for (let i = sorted.length - 1; i >= 0; i--) {
        if (this.iou(candidate, sorted[i]) > iouThreshold) {
          sorted.splice(i, 1);
        }
      }
    }

    return result;
  }

  iou(boxA, boxB) {
    const xA = Math.max(boxA.x1, boxB.x1);
    const yA = Math.max(boxA.y1, boxB.y1);
    const xB = Math.min(boxA.x2, boxB.x2);
    const yB = Math.min(boxA.y2, boxB.y2);
    const interW = Math.max(0, xB - xA);
    const interH = Math.max(0, yB - yA);
    const interArea = interW * interH;
    const boxAArea = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1);
    const boxBArea = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1);
    const union = boxAArea + boxBArea - interArea;

    return union <= 0 ? 0 : interArea / union;
  }

  async run(videoElement, canvasElement) {
    if (!this.session) return [];

    const inputTensor = this.preprocess(videoElement);
    const feeds = { [this.inputName]: inputTensor };
    const results = await this.session.run(feeds);
    const outputName = this.session.outputNames[0];
    const output = results[outputName];

    return this.postprocess(output, canvasElement.width, canvasElement.height);
  }
}