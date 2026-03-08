# 🚀 QUICK START GUIDE

## Installation en 5 minutes

### 1. Créer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate pour Windows
```

### 2. Installer PyTorch avec CUDA 11.8 (GTX 1650)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Tester l'installation

```bash
python test_installation.py
```

Si tout est ✅, vous êtes prêt !

---

## Test rapide (Mode Heuristique - Sans entraînement)

### Option A : Avec webcam

```bash
# Dans config.py, vérifier :
VIDEO_SOURCE = 0  # Webcam
USE_HEURISTIC = True

# Lancer :
python main_behavioral.py
```

### Option B : Avec une vidéo

```bash
# Dans config.py, modifier :
VIDEO_SOURCE = "path/to/your/video.mp4"
USE_HEURISTIC = True

# Lancer :
python main_behavioral.py
```

**Le système va :**

- Détecter les personnes avec YOLO
- Les tracker avec DeepSORT
- Extraire leur pose avec MediaPipe
- Calculer un score heuristique basé sur les gestes suspects
- Afficher les alertes en temps réel

**Touches :**

- **Q** : Quitter
- **S** : Sauvegarder frame

---

## Entraînement du modèle (Mode LSTM - Meilleure précision)

### 1. Préparer vos vidéos

```
project/
├── videos/
│   ├── shoplifting/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── normal/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
```

**Important :**

- Minimum 20 vidéos shoplifting + 20 normales
- Idéalement 50+ de chaque pour de bons résultats
- Durée : 30 secondes à 2 minutes par vidéo

### 2. Lancer l'entraînement

```bash
python train_model.py \
    --shoplifting_dir videos/shoplifting \
    --normal_dir videos/normal \
    --output model_shoplifting.pth \
    --epochs 50
```

**Temps estimé sur GTX 1650 :**

- 10 vidéos : ~5-10 minutes
- 50 vidéos : ~20-30 minutes
- 100 vidéos : ~40-60 minutes

### 3. Activer le modèle entraîné

```python
# Dans config.py, modifier :
USE_HEURISTIC = False
USE_LSTM = True
MODEL_PATH_LSTM = "model_shoplifting.pth"
```

### 4. Lancer avec le modèle LSTM

```bash
python main_behavioral.py
```

---

## Comprendre les résultats

### Scores heuristiques (mode baseline)

```
Score 0-39  → ✓ Normal (vert)
Score 40-69 → ⚡ Suspect (orange)
Score 70+   → 🚨 Risque élevé (rouge)
```

### Probabilités LSTM (après entraînement)

```
0-64%   → ✓ Normal (vert)
65-84%  → ⚡ Suspect (orange)
85-100% → 🚨 Risque élevé (rouge)
```

### HUD en temps réel

```
FPS: 25.3              ← Performance système
Personnes: 4           ← Nombre de personnes trackées
🚨 Risque élevé: 1    ← Alerte critique
⚡ Suspect: 0          ← Alerte modérée
```

---

## Ajuster les paramètres (config.py)

### Pour améliorer la détection

```python
CONFIDENCE_THRESHOLD = 0.4  # Plus permissif (défaut: 0.5)
```

### Pour réduire les faux positifs

```python
CLASSIFIER_THRESHOLD = 0.75  # Plus strict (défaut: 0.65)
ALERT_THRESHOLD_HIGH = 0.90  # Alertes moins fréquentes
```

### Pour améliorer les performances (FPS)

```python
INFERENCE_SIZE = 480  # Plus petit (défaut: 640)
BEHAVIOR_WINDOW_SIZE = 60  # Historique plus court
```

---

## Problèmes courants

### ❌ "CUDA out of memory"

```python
# config.py
LSTM_HIDDEN_DIM = 32  # Réduire (défaut: 64)
```

### ❌ FPS trop bas (<15)

```python
# config.py
INFERENCE_SIZE = 480
MODEL_PATH = "yolo26n.pt"  # Vérifier que c'est bien 'n' (nano)
```

### ❌ Trop de faux positifs

```python
# config.py
CLASSIFIER_THRESHOLD = 0.80
ALERT_THRESHOLD_MEDIUM = 0.75
```

### ❌ MediaPipe ne détecte pas la pose

- La personne doit être suffisamment visible
- Éviter les occlusions importantes
- Vérifier l'éclairage de la vidéo

---

## Workflow recommandé

### Phase 1 : Tester le système (1h)

1. ✅ Installer les dépendances
2. ✅ Tester avec webcam en mode heuristique
3. ✅ Ajuster les paramètres de visualisation

### Phase 2 : Collecter des données (1-2 jours)

1. ✅ Récupérer 50+ vidéos shoplifting
2. ✅ Récupérer 50+ vidéos normales
3. ✅ Organiser dans `videos/shoplifting` et `videos/normal`

### Phase 3 : Entraîner le modèle (2-3h)

1. ✅ Lancer `train_model.py`
2. ✅ Surveiller train/val accuracy
3. ✅ Tester avec des vidéos jamais vues

### Phase 4 : Déployer (30min)

1. ✅ Activer `USE_LSTM = True`
2. ✅ Tester en conditions réelles
3. ✅ Affiner les seuils d'alerte

---

## Performances attendues

### Mode Heuristique (baseline)

- **Précision** : 65-75%
- **FPS** : 25-30
- **Avantages** : Immédiat, pas d'entraînement
- **Inconvénients** : Plus de faux positifs

### Mode LSTM (après entraînement)

- **Précision** : 80-92% (selon dataset)
- **FPS** : 20-28
- **Avantages** : Meilleure précision, moins de faux positifs
- **Inconvénients** : Nécessite données + entraînement

---

## Fichiers du projet

```
📁 Fichiers essentiels
├── config.py              ← Configuration (À MODIFIER)
├── main_behavioral.py     ← Point d'entrée principal
├── train_model.py         ← Entraînement LSTM
├── behavior_analyzer.py   ← Analyse pose + gestes
├── behavior_classifier.py ← Modèle LSTM + heuristique
├── detector.py            ← YOLO détection
├── tracker.py             ← DeepSORT tracking
└── utils.py               ← Visualisation

📁 Fichiers de support
├── requirements.txt       ← Dépendances
├── README.md             ← Documentation complète
├── QUICK_START.md        ← Ce fichier
└── test_installation.py  ← Test installation
```

---

## Support & Documentation

- **README.md** : Documentation complète
- **config.py** : Tous les paramètres commentés
- **Architecture** : Voir schéma dans README.md

---

**🎯 Objectif : Détecter les vols à l'étalage en temps réel avec 85%+ précision**

**GTX 1650 optimisé | 20-30 FPS | Latence <2 sec**
