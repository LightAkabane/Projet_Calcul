import { REID_WIDTH, REID_HEIGHT, REID_MEAN, REID_STD } from './config.js';

export class ReidEngine {
  constructor() {
    this.session = null;
    this.inputName = null;
    this.outputName = null;
    this.canvas = document.createElement('canvas');
    this.canvas.width = REID_WIDTH;
    this.canvas.height = REID_HEIGHT;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
  }

  async initialize() {
    try {
      this.session = await ort.InferenceSession.create('model/resnet18_reid.onnx', {
        executionProviders: ['webgpu']
      });
      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];
      console.log('Session OSNet Re-ID initialisée');
      return true;
    } catch (e) {
      console.warn('OSNet Re-ID non initialisé:', e?.message);
      return false;
    }
  }

  preprocess(detection, videoElement) {
    this.ctx.clearRect(0, 0, REID_WIDTH, REID_HEIGHT);
    this.ctx.drawImage(
      videoElement,
      detection.x, detection.y, detection.w, detection.h,
      0, 0, REID_WIDTH, REID_HEIGHT
    );

    const imageData = this.ctx.getImageData(0, 0, REID_WIDTH, REID_HEIGHT);
    const data = imageData.data;
    const float32Data = new Float32Array(3 * REID_WIDTH * REID_HEIGHT);
    const size = REID_WIDTH * REID_HEIGHT;

    for (let i = 0; i < size; i++) {
      const r = data[4*i] / 255;
      const g = data[4*i+1] / 255;
      const b = data[4*i+2] / 255;
      float32Data[i] = (r - REID_MEAN[0]) / REID_STD[0];
      float32Data[i+size] = (g - REID_MEAN[1]) / REID_STD[1];
      float32Data[i+2*size] = (b - REID_MEAN[2]) / REID_STD[2];
    }

    return new ort.Tensor('float32', float32Data, [1, 3, REID_HEIGHT, REID_WIDTH]);
  }

  normalize(embedding) {
    let norm = 0;
    for (let i = 0; i < embedding.length; i++) {
      norm += embedding[i] * embedding[i];
    }
    norm = Math.sqrt(norm) || 1;

    const normalized = new Float32Array(embedding.length);
    for (let i = 0; i < embedding.length; i++) {
      normalized[i] = embedding[i] / norm;
    }

    return normalized;
  }

  async extract(detection, videoElement) {
    if (!this.session) return null;

    try {
      const inputTensor = this.preprocess(detection, videoElement);
      const feeds = { [this.inputName]: inputTensor };
      const results = await this.session.run(feeds);
      const output = results[this.outputName];
      const embedding = output.data;

      return this.normalize(embedding);
    } catch (e) {
      console.error('Erreur extraction Re-ID:', e);
      return null;
    }
  }

  cosineSimilarity(a, b) {
    let dot = 0, na = 0, nb = 0;

    for (let i = 0; i < a.length; i++) {
      const va = a[i], vb = b[i];
      dot += va * vb;
      na += va * va;
      nb += vb * vb;
    }

    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }
}