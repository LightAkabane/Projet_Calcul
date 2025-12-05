import { UIManager } from './ui.js';
import { CameraManager } from './camera.js';
import { DetectionEngine } from './detection.js';
import { ReidEngine } from './reid.js';
import { TrackingEngine } from './tracking.js';
import { AudioManager } from './audio.js';
import { Renderer } from './rendering.js';

class CrowdlyApp {
  constructor() {
    this.ui = new UIManager();
    this.camera = new CameraManager(
      document.getElementById('video'),
      document.getElementById('overlay')
    );
    this.detectionEngine = new DetectionEngine();
    this.reidEngine = new ReidEngine();
    this.trackingEngine = new TrackingEngine(this.reidEngine);
    this.audioManager = new AudioManager(
      document.getElementById('lofiAudio'),
      this.ui
    );
    this.renderer = new Renderer(document.getElementById('overlay'));
    this.isRunning = false;
  }

  async initialize() {
    try {
      this.ui.setStatus('Demande d\'accès à la caméra...');
      await this.camera.initialize();

      this.ui.setStatus('Initialisation WebGPU / modèles...');
      await this.detectionEngine.initialize();
      const reidReady = await this.reidEngine.initialize();

      // Slider confiance
      this.ui.getThresholdInput().addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        this.detectionEngine.setConfidenceThreshold(value);
        this.ui.setThreshold(value);
      });

      this.ui.setStatus('Modèles chargés, détection en cours...');
      this.isRunning = true;
      this.detectionLoop();
    } catch (err) {
      console.error(err);
      this.ui.setStatus('Erreur initialisation : ' + err.message);
    }
  }

  async detectionLoop() {
    if (!this.isRunning) return;

    // Détection YOLO
    let detections = await this.detectionEngine.run(
      this.camera.getVideoElement(),
      this.camera.getCanvasElement()
    );

    // Extraction Re-ID
    if (this.reidEngine.session) {
      for (const det of detections) {
        det.embedding = await this.reidEngine.extract(
          det,
          this.camera.getVideoElement()
        );
      }
    }

    // Suivi
    detections = this.trackingEngine.updateTracks(detections);
    const activeDetections = this.trackingEngine.getConfirmedDetections(detections);

    // Affichage
    this.renderer.drawDetections(activeDetections);

    // Mise à jour UI
    this.ui.updateDetectionCount(activeDetections.length);
    this.ui.updateUniqueCount(this.trackingEngine.getUniqueIdentitiesCount());

    requestAnimationFrame(() => this.detectionLoop());
  }

  stop() {
    this.isRunning = false;
    this.camera.stop();
    this.audioManager.stop();
  }
}

// Lancer l'app au chargement
window.addEventListener('DOMContentLoaded', () => {
  const app = new CrowdlyApp();
  app.initialize();
});