export class UIManager {
  constructor() {
    this.statusEl = document.getElementById('status');
    this.counterEl = document.getElementById('counter');
    this.uniqueCounterEl = document.getElementById('unique-counter');
    this.threshEl = document.getElementById('thresh');
    this.threshVal = document.getElementById('threshVal');
    this.vinyl = document.getElementById('vinyl');
    this.btnPlay = document.getElementById('btnPlay');
    this.btnMute = document.getElementById('btnMute');
  }

  setStatus(message) {
    this.statusEl.textContent = message;
  }

  updateDetectionCount(count) {
    this.counterEl.textContent = `👀 Personnes détectées : ${count}`;
  }

  updateUniqueCount(count) {
    this.uniqueCounterEl.textContent = `🌟 Personnes uniques vues : ${count}`;
  }

  setThreshold(value) {
    this.threshVal.textContent = value.toFixed(2);
  }

  setPlayingState(playing) {
    this.vinyl.classList.toggle('spin', playing);
    this.btnPlay.textContent = playing ? '⏸︎ Pause lofi' : '▶︎ Play lofi';
  }

  setMutedState(muted) {
    this.btnMute.textContent = muted ? '🔊 Unmute' : '🔇 Mute';
  }

  getThresholdInput() {
    return this.threshEl;
  }

  getPlayButton() {
    return this.btnPlay;
  }

  getMuteButton() {
    return this.btnMute;
  }
}