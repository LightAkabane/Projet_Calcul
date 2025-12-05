import { MAX_MISSED_FRAMES, MIN_HITS, IOU_MATCH_THRESHOLD, IDENTITY_THRESHOLD } from './config.js';

export class TrackingEngine {
  constructor(reidEngine) {
    this.reidEngine = reidEngine;
    this.tracks = [];
    this.identities = [];
    this.nextTrackId = 1;
    this.nextIdentityId = 1;
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

  assignIdentityToTrack(track) {
    if (!track.embedding) return;

    let bestIdentity = null, bestSim = -1;

    for (const ident of this.identities) {
      const sim = this.reidEngine.cosineSimilarity(track.embedding, ident.embedding);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdentity = ident;
      }
    }

    if (bestIdentity && bestSim >= IDENTITY_THRESHOLD) {
      track.identityId = bestIdentity.id;
      const emb = bestIdentity.embedding;
      const newEmb = track.embedding;

      // EMA (Exponential Moving Average)
      for (let i = 0; i < emb.length; i++) {
        emb[i] = 0.7 * newEmb[i] + 0.3 * emb[i];
      }
    } else {
      const id = this.nextIdentityId++;
      track.identityId = id;
      this.identities.push({
        id,
        embedding: track.embedding.slice()
      });
    }
  }

  updateTracks(detections) {
    this.tracks.forEach(t => { t.matched = false; });

    detections.forEach(det => {
      const detBox = {
        x1: det.x, y1: det.y,
        x2: det.x + det.w, y2: det.y + det.h
      };

      let bestTrack = null, bestIoU = 0;

      this.tracks.forEach(track => {
        const trackBox = {
          x1: track.x, y1: track.y,
          x2: track.x + track.w, y2: track.y + track.h
        };
        const iouVal = this.iou(detBox, trackBox);

        if (iouVal > bestIoU) {
          bestIoU = iouVal;
          bestTrack = track;
        }
      });

      if (bestTrack && bestIoU > IOU_MATCH_THRESHOLD) {
        // Mise à jour de la piste existante (EMA)
        const alpha = 0.7;
        bestTrack.x = alpha * det.x + (1 - alpha) * bestTrack.x;
        bestTrack.y = alpha * det.y + (1 - alpha) * bestTrack.y;
        bestTrack.w = alpha * det.w + (1 - alpha) * bestTrack.w;
        bestTrack.h = alpha * det.h + (1 - alpha) * bestTrack.h;

        if (det.embedding) {
          bestTrack.embedding = det.embedding.slice();
        }

        bestTrack.missed = 0;
        bestTrack.hits = (bestTrack.hits || 0) + 1;
        bestTrack.matched = true;

        if (!bestTrack.confirmed && bestTrack.hits >= MIN_HITS) {
          bestTrack.confirmed = true;
          this.assignIdentityToTrack(bestTrack);
        }

        det.id = bestTrack.confirmed ? bestTrack.id : null;
      } else {
        // Créer une nouvelle piste
        const newTrack = {
          id: this.nextTrackId++,
          x: det.x, y: det.y,
          w: det.w, h: det.h,
          missed: 0, hits: 1,
          confirmed: false,
          matched: true,
          embedding: det.embedding ? det.embedding.slice() : null,
          identityId: null
        };

        this.tracks.push(newTrack);
        det.id = null;
      }
    });

    // Nettoyer les pistes perdues
    this.tracks = this.tracks.filter(track => {
      if (!track.matched) {
        track.missed = (track.missed || 0) + 1;
      }
      return track.missed <= MAX_MISSED_FRAMES;
    });

    return detections;
  }

  getConfirmedDetections(detections) {
    return detections.filter(det => det.id != null);
  }

  getUniqueIdentitiesCount() {
    return this.identities.length;
  }

  reset() {
    this.tracks = [];
    this.identities = [];
    this.nextTrackId = 1;
    this.nextIdentityId = 1;
  }
}