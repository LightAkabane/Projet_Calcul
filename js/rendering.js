export class Renderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
  }

  drawDetections(detections) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.lineWidth = 2;
    this.ctx.font = '600 14px DM Sans, system-ui';

    detections.forEach(det => {
      this.ctx.strokeStyle = '#34d399';
      this.ctx.fillStyle = '#a78bfa';
      this.ctx.shadowColor = 'rgba(167,139,250,0.45)';
      this.ctx.shadowBlur = 12;

      this.ctx.strokeRect(det.x, det.y, det.w, det.h);

      const label = `Person ${(det.score * 100).toFixed(1)}%`;
      const textX = det.x + 6;
      const textY = Math.max(16, det.y - 8);

      this.ctx.fillText(label, textX, textY);
      this.ctx.shadowBlur = 0;
    });
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}