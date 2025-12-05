export class AudioManager {
  constructor(audioElement, uiManager) {
    this.audio = audioElement;
    this.uiManager = uiManager;
    this.setupEventListeners();
  }

  setupEventListeners() {
    this.uiManager.getPlayButton().addEventListener('click', () => this.togglePlay());
    this.uiManager.getMuteButton().addEventListener('click', () => this.toggleMute());
  }

  async togglePlay() {
    try {
      if (this.audio.paused) {
        await this.audio.play();
        this.uiManager.setPlayingState(true);
      } else {
        this.audio.pause();
        this.uiManager.setPlayingState(false);
      }
    } catch (e) {
      console.warn('Lecture audio bloquée par le navigateur');
    }
  }

  toggleMute() {
    this.audio.muted = !this.audio.muted;
    this.uiManager.setMutedState(this.audio.muted);
  }

  stop() {
    this.audio.pause();
    this.audio.currentTime = 0;
  }
}