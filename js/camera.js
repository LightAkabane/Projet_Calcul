export class CameraManager {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.stream = null;
  }

  async initialize(width = 640, height = 480) {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width, height }
      });
      this.video.srcObject = this.stream;

      return new Promise((resolve, reject) => {
        this.video.onloadedmetadata = () => {
          this.video.play();
          this.canvas.width = this.video.videoWidth;
          this.canvas.height = this.video.videoHeight;
          resolve({ width: this.canvas.width, height: this.canvas.height });
        };
        this.video.onerror = reject;
      });
    } catch (err) {
      throw new Error(`Erreur accès caméra : ${err.message}`);
    }
  }

  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }

  getVideoElement() {
    return this.video;
  }

  getCanvasElement() {
    return this.canvas;
  }
}