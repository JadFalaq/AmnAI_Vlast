# ============================================================
# config.py — Configuration centrale du projet
# Modifier ici pour changer les paramètres sans toucher au code
# ============================================================

# ------------------------------------------------------------
# 📷 SOURCE VIDÉO
# ------------------------------------------------------------
# Mettre 0 pour webcam, ou le chemin vers une vidéo :
#   VIDEO_SOURCE = 0                    → webcam
#   VIDEO_SOURCE = "video_test.mp4"     → fichier vidéo
# ------------------------------------------------------------
VIDEO_SOURCE = r"C:\Users\falaq\OneDrive\Desktop\Jad Falaq\AmnAI\Shop DataSet\shop lifters\shop_lifter_78.mp4"

# ------------------------------------------------------------
# 🤖 MODÈLE YOLO26
# ------------------------------------------------------------
MODEL_PATH = "yolo26n.pt"       # nano = le plus rapide pour GTX 1650
                                # options : yolo26n / yolo26s / yolo26m
CONFIDENCE_THRESHOLD = 0.5      # Seuil de confiance détection (0-1)
DEVICE = "cuda"                 # "cuda" pour GPU, "cpu" pour processeur
INFERENCE_SIZE = 640            # Taille image YOLO (640 = standard)
NMS_IOU = 0.6

# ------------------------------------------------------------
# 👤 ON NE DÉTECTE QUE LES PERSONNES (classe 0 du COCO)
# ------------------------------------------------------------
DETECT_CLASS_PERSON = 0

# ------------------------------------------------------------
# 🔍 DEEPSORT — Paramètres du tracker
# ------------------------------------------------------------
# max_age : frames où une personne peut DISPARAÎTRE avant que
#           son ID soit supprimé. À ~30fps :
#               60  = ~2 secondes
#               90  = ~3 secondes  ← RECOMMANDÉ
#               150 = ~5 secondes
# ------------------------------------------------------------
DEEPSORT_MAX_AGE = 90

# n_init : détections nécessaires avant de CONFIRMER un track.
#          3 = par défaut, évite les faux positifs fugitifs.
# ------------------------------------------------------------
DEEPSORT_N_INIT = 3

# max_cosine_distance : seuil de similarité d'apparence.
#   0.2  = strict   (doit ressembler beaucoup)
#   0.4  = équilibre ← RECOMMANDÉ
#   0.6  = très permissif
# ------------------------------------------------------------
DEEPSORT_MAX_COSINE_DIST = 0.4

# ------------------------------------------------------------
# 🎨 COULEURS (BGR pour OpenCV)
# ------------------------------------------------------------
COLOR_CONFIRMED   = (0, 255, 0)          # Vert  → personne confirmée
COLOR_UNCONFIRMED = (150, 150, 150)      # Gris  → en cours de confirmation
COLOR_FPS         = (255, 255, 255)      # Blanc → texte HUD

# ------------------------------------------------------------
# 💾 SAUVEGARDE
# ------------------------------------------------------------
SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = "output_tracking.avi"

# ------------------------------------------------------------
# 🧠 ANALYSE COMPORTEMENTALE
# ------------------------------------------------------------
# Taille de la fenêtre pour accumuler l'historique (en frames)
BEHAVIOR_WINDOW_SIZE = 90  # 3 secondes à 30fps

# Seuil de vitesse pour considérer une personne stationnaire
STATIONARY_THRESHOLD = 1.0  # pixels par frame

# ------------------------------------------------------------
# 🤖 CLASSIFICATEUR LSTM
# ------------------------------------------------------------
# Longueur de séquence pour le LSTM (nombre de frames)
SEQUENCE_LENGTH = 60  # 2 secondes à 30fps

# Dimension du vecteur de features par frame
FEATURE_DIM = 10  # 5 mouvement + 4 gestes + 1 total

# Architecture LSTM
LSTM_HIDDEN_DIM = 64  # Réduit pour GTX 1650
LSTM_LAYERS = 2
DROPOUT_RATE = 0.4

# Seuil de classification
CLASSIFIER_THRESHOLD = 0.65  # Probabilité > 0.65 → shoplifting


# Mode de détection
USE_HEURISTIC = False
USE_LSTM = True
MODEL_PATH_LSTM = r"C:\Users\falaq\OneDrive\Desktop\Jad Falaq\AmnAI\AmnAi_Vlast\model_shoplifting.pth"
# ------------------------------------------------------------
# 🚨 ALERTES
# ------------------------------------------------------------
# Seuils d'alerte
ALERT_THRESHOLD_HIGH = 0.5  # Critique
ALERT_THRESHOLD_MEDIUM = 0.4  # Suspect

# Couleurs alertes (BGR)
COLOR_HIGH_RISK = (0, 0, 255)  # Rouge
COLOR_MEDIUM_RISK = (0, 165, 255)  # Orange
COLOR_NORMAL = (0, 255, 0)  # Vert
