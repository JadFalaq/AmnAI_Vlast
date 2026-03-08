# 🛒 Shoplifting Detection — Analyse comportementale en temps réel

Système complet de détection de vol à l’étalage basé sur l’analyse du comportement humain. Le pipeline combine détection, tracking, pose et classification temporelle.

## 🎯 Objectif du projet

- Détecter des comportements suspects en temps réel
- Fonctionner sur caméra de surveillance standard
- Permettre un mode heuristique (sans entraînement) et un mode LSTM (avec entraînement)

## 🧱 Stack & modèles

- Python 3.10+
- PyTorch 2.x + CUDA 11.8
- OpenCV
- Ultralytics YOLO (modèle YOLO26n)
- DeepSORT (tracking)
- MediaPipe Pose (pose humaine)
- LSTM bidirectionnel + attention

## 🧭 Pipeline complet

```
Caméra/vidéo
   ↓
YOLO26n (détection personnes)
   ↓
DeepSORT (ID unique par personne)
   ↓
MediaPipe Pose (landmarks)
   ↓
Behavior Analyzer (features comportementales)
   ↓
Classifier (LSTM ou heuristique)
   ↓
Alertes + Overlay en temps réel
```

## 📦 Structure du projet

```
AmnAi_Vlast/
├── config.py
├── detector.py
├── tracker.py
├── behavior_analyzer.py
├── behavior_classifier.py
├── utils.py
├── main.py
├── main_behavioral.py
├── train_model.py
├── demo.py
├── visualize_features.py
├── test_installation.py
├── requirements.txt
└── README.md
```

## ✅ Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 🧪 Vérifier l’installation

```bash
python test_installation.py
```

## � Préparer les vidéos

Deux dossiers séparés :

```
videos/
  shoplifting/
  normal/
```

Formats supportés : .mp4, .avi, .mov  
Durée conseillée : 30s à 2min par vidéo.

## 🚀 Démarrage rapide (mode heuristique)

Dans config.py :

```
USE_HEURISTIC = True
USE_LSTM = False
MODEL_PATH_LSTM = None
```

Puis :

```bash
python main_behavioral.py
```

## 🏋️ Entraîner le LSTM

```bash
python train_model.py \
  --shoplifting_dir "chemin/vers/videos/shoplifting" \
  --normal_dir "chemin/vers/videos/normal" \
  --output "model_shoplifting.pth" \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.001
```

À la fin, le script imprime l’accuracy sur le set de test et sauvegarde le meilleur modèle.

## ▶️ Lancer en mode LSTM

Dans config.py :

```
USE_HEURISTIC = False
USE_LSTM = True
MODEL_PATH_LSTM = "model_shoplifting.pth"
```

Puis :

```bash
python main_behavioral.py
```

## 🧩 Scripts utiles

- Détection + tracking : `python main.py`
- Pipeline complet : `python main_behavioral.py`
- Entraînement LSTM : `python train_model.py ...`
- Démo rapide : `python demo.py --video path/to/video.mp4`
- Visualiser les features : `python visualize_features.py --video path/to/video.mp4`

## 🧠 Features comportementales extraites

- Mouvement : vitesse moyenne, écart-type, vitesse max, temps stationnaire
- Gestes : main → poche, main → sac, corps penché, regard furtif
- Total des gestes cumulés

## �️ Paramètres clés (config.py)

```
CONFIDENCE_THRESHOLD
INFERENCE_SIZE
DEEPSORT_MAX_AGE
BEHAVIOR_WINDOW_SIZE
SEQUENCE_LENGTH
CLASSIFIER_THRESHOLD
ALERT_THRESHOLD_HIGH
ALERT_THRESHOLD_MEDIUM
```

## 📊 Performances attendues (GTX 1650)

- Heuristique : 25–30 FPS, latence 0s
- LSTM : 20–28 FPS, latence ~2s

## �️ Dépannage rapide

- FPS bas : réduire INFERENCE_SIZE (ex: 480)
- Peu de séquences extraites : baisser CONFIDENCE_THRESHOLD
- Trop de faux positifs : augmenter CLASSIFIER_THRESHOLD

## � Bonnes pratiques

- Équilibrer les classes (autant de normal que de shoplifting)
- Varier les conditions de lumière et d’angle
- Éviter les vidéos trop courtes pour le LSTM

## 📄 Licence

Usage éducatif uniquement.
