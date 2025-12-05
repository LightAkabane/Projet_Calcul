# 🎵 Crowdly - Détection & Suivi de Personnes avec WebGPU

Crowdly est une application web de **détection et suivi en temps réel** de personnes utilisant **WebGPU** pour l'accélération GPU directement dans le navigateur.

 [Démo](https://lightakabane.github.io/Projet_Calcul/) 

## ✨ Fonctionnalités

- 👁️ **Détection de personnes** en temps réel via YOLOv8n
- 🔄 **Suivi multi-cible** avec matching IoU et EMA
- 🌟 **Re-identification** (Re-ID) avec OSNet ResNet18
- 📊 **Compteurs** : personnes détectées + personnes uniques vues
- 🎚️ **Slider de confiance** dynamique
- 🎵 **Ambient lofi** avec boutons Play/Mute
- 🎨 **Design glassmorphism** moderne avec thème dark
- ⚡ **Exécution sur GPU** via WebGPU (accélération matérielle)

## 🏗️ Architecture Modulaire

Le projet suit une architecture **propre et professionnelle** avec séparation des responsabilités :

```
crowdly/
├── index.html                 # Point d'entrée HTML
├── css/
│   └── styles.css            # Tous les styles (glassmorphism + animations)
├── media/
│   ├── 200w.gif           
│   ├── 200w2.gif             
│   └── song.mp3   
├── model/
│   ├── resnet18_reid.onnx          
│   └── yolov8n.onnx   
├── python/
│   ├── get_resnet_onnx.py          
│   └── get_yolov8_onnx.py   
├── js/
│   ├── app.js                # Point d'entrée JS - orchestration principale
│   ├── config.js             # Constantes globales
│   ├── ui.js                 # Gestion de l'interface utilisateur
│   ├── camera.js             # Gestion de la caméra (getUserMedia)
│   ├── detection.js          # Moteur YOLO (détection)
│   ├── reid.js               # Moteur Re-ID (extraction de features)
│   ├── tracking.js           # Engine de suivi (tracking + identités)
│   ├── audio.js              # Gestion audio (play/mute)
│   └── rendering.js          # Affichage des résultats (canvas)

```

### 📦 Modules

| Module | Responsabilité |
|--------|---|
| **app.js** | Orchestration, initialisation, boucle principale |
| **config.js** | Constantes (seuils, dimensions, hyperparamètres) |
| **ui.js** | UIManager - interaction avec le DOM |
| **camera.js** | CameraManager - gestion flux vidéo |
| **detection.js** | DetectionEngine - YOLO + NMS |
| **reid.js** | ReidEngine - extraction features + similarity |
| **tracking.js** | TrackingEngine - suivi + identités uniques |
| **audio.js** | AudioManager - play/mute lofi |
| **rendering.js** | Renderer - dessin canvas |

## 🚀 Démarrage Rapide

### Prérequis

- **Navigateur moderne** : Chrome/Edge 113+ (WebGPU support)
- **GPU** : Qualsiasi GPU supporté (NVIDIA, AMD, Intel)
- **Caméra web** accessible

### Installation

1. **Cloner/télécharger le projet**
```bash
git clone <repo-url>
cd crowdly
```

2. **Télécharger les modèles ONNX**

   Créer un dossier `models/` et télécharger via les fichiers pythons mis dans le dossier `python/`:
   - [YOLOv8n.onnx](https://github.com/LightAkabane/Projet_Calcul/tree/main/python/get_yolov8_onnx.py)
   - [ResNet18 Re-ID](https://github.com/LightAkabane/Projet_Calcul/tree/main/python/get_resnet_onnx.py)


3. **Démarrer un serveur local**

   ```bash
   # Avec Python 3
   python -m http.server 8000
   
   # Ou avec Node.js
   npx http-server
   
   # Ou avec PHP
   php -S localhost:8000
   ```

4. **Ouvrir dans le navigateur**
   ```
   http://localhost:8000
   ```

### ⚙️ Configuration

Modifier les constantes dans `js/config.js` :

```javascript
// Dimensions du modèle
export const MODEL_WIDTH = 640;
export const MODEL_HEIGHT = 640;

// Suivi
export const MAX_MISSED_FRAMES = 20;      // Frames avant suppression piste
export const MIN_HITS = 5;                // Hits avant confirmation
export const IOU_MATCH_THRESHOLD = 0.3;   // IoU min pour match
export const IDENTITY_THRESHOLD = 0.73;   // Similarité Re-ID

// Re-ID
export const REID_WIDTH = 128;
export const REID_HEIGHT = 256;
```

## 🎮 Utilisation

### Contrôles UI

| Élément | Fonction |
|---------|----------|
| **▶︎ Play lofi** | Démarrer/pausser la musique |
| **🔇 Mute** | Couper/rétablir le son |
| **Slider Confiance** | Ajuster le seuil de confiance YOLO (0.2 → 0.95) |
| **Canvas** | Affichage des détections en temps réel |
| **Compteurs** | Personnes détectées + uniques vues |

### Indicateurs

- 🟢 **Boîtes vertes** = Détections YOLO
- 💜 **Labels lilas** = Score de confiance
- 📊 **Statistiques bas** = Compteurs live

## 🔧 Architecture Détaillée

### Flux de Traitement

```
Video Stream
    ↓
[DetectionEngine] → YOLO (YOLOv8n)
    ↓
[ReidEngine] → Extract features (ResNet18)
    ↓
[TrackingEngine] → Match tracks + Assign identities
    ↓
[Renderer] → Draw on Canvas
    ↓
UI Update (counters + status)
```

### Algorithmes Clés

**1. Détection : YOLOv8n**
- Input : Frame vidéo (640×640)
- Output : Bounding boxes + scores
- Post-processing : Non-Maximum Suppression (NMS)

**2. Re-ID : ResNet18**
- Input : Crop détection (128×256)
- Output : Feature embedding (512-dim)
- Normalisation : L2 + moyenne mobile (EMA)

**3. Suivi : IoU Matching + EMA**
- Match : Intersection over Union (IoU > 0.3)
- Update : Exponential Moving Average (α=0.7)
- Confirmation : MIN_HITS = 5 frames

**4. Identités Uniques**
- Comparaison : Cosine similarity (threshold = 0.73)
- Update : EMA sur embeddings
- Persistance : Long-term (survit au hors-champ)

## 📈 Performance

| Métrique | Valeur |
|----------|--------|
| **FPS** | 15-30 FPS (RTX 3070+) |
| **Latence** | ~50-100ms/frame |
| **RAM** | ~300-500 MB |
| **VRAM** | ~1-2 GB |

*Note: Dépend du GPU et de la taille vidéo*

## 🐛 Troubleshooting

### ❌ "WebGPU non disponible"
- **Solution** : Utiliser Chrome/Edge 113+ ou Edge Insider
- Vérifier : `chrome://gpu` → WebGPU doit être en vert

### ❌ "Impossible charger les modèles ONNX"
- **Solution** : Vérifier l'URL des `.onnx` dans `detection.js` et `reid.js`
- Les fichiers doivent être accessibles via HTTP(S)

### ❌ "getUserMedia erreur"
- **Solution** : La caméra doit être autorisée (check permissions navigateur)
- Tester d'abord : `about:preferences#privacy` → Caméra

### ❌ "Erreur: Cannot use import outside a module"
- **Solution** : Vérifier que l'HTML a `<script type="module" src="js/app.js"></script>`
- Ne pas charger d'autres scripts directement !

## 🎨 Customization

### Modifier les couleurs

Éditer `css/styles.css` (variables CSS) :

```css
:root {
  --bg-1: #0f1020;           /* Fond principal */
  --accent: #a78bfa;          /* Accent (lilas) */
  --accent-2: #34d399;        /* Accent 2 (mint) */
  --text: #e7e7ff;            /* Texte */
}
```

### Modifier la musique

Remplacer `song.mp3` par votre propre fichier lofi, puis mettre à jour `index.html` :

```html
<audio id="lofiAudio" loop crossorigin="anonymous">
  <source src="votre-musique.mp3" type="audio/mpeg" />
</audio>
```

### Modifier les images latérales

Remplacer `200w.gif` et `200w2.gif` par vos propres images (aspect ratio 4:5 recommandé)

## 📚 Ressources

- [ONNX Runtime Web Docs](https://onnxruntime.ai/docs/get-started/with-javascript/)
- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [OSNet Re-ID Paper](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)

## 📝 Licence

MIT - Libre d'utilisation et de modification

## 👤 Auteur

- Benosmane Yacine
- Benmouloud Mehdi

---

**⚡ Made with WebGPU × ONNX Runtime × Modern Web Stack**
