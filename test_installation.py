# ============================================================
# test_installation.py – Test rapide de l'installation
# ============================================================
# Vérifie que toutes les dépendances sont installées correctement
# Usage : python test_installation.py
# ============================================================

import sys
from colorama import Fore, Style, init

init(autoreset=True)

print("\n" + "="*60)
print("🔍 TEST D'INSTALLATION - Shoplifting Detection")
print("="*60 + "\n")

# Test 1 : PyTorch + CUDA
print(f"{Fore.CYAN}[1/6] Test PyTorch...{Style.RESET_ALL}")
try:
    import torch
    print(f"   ✅ PyTorch version : {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   ✅ CUDA disponible : {torch.cuda.get_device_name(0)}")
        print(f"   ✅ VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"   ⚠️  CUDA non disponible (mode CPU uniquement)")
except Exception as e:
    print(f"   ❌ Erreur : {e}")
    sys.exit(1)

# Test 2 : OpenCV
print(f"\n{Fore.CYAN}[2/6] Test OpenCV...{Style.RESET_ALL}")
try:
    import cv2
    print(f"   ✅ OpenCV version : {cv2.__version__}")
except Exception as e:
    print(f"   ❌ Erreur : {e}")
    sys.exit(1)

# Test 3 : Ultralytics (YOLO)
print(f"\n{Fore.CYAN}[3/6] Test Ultralytics (YOLO)...{Style.RESET_ALL}")
try:
    from ultralytics import YOLO
    print(f"   ✅ Ultralytics installé")
except Exception as e:
    print(f"   ❌ Erreur : {e}")
    sys.exit(1)

# Test 4 : DeepSORT
print(f"\n{Fore.CYAN}[4/6] Test DeepSORT...{Style.RESET_ALL}")
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    print(f"   ✅ DeepSORT installé")
except Exception as e:
    print(f"   ❌ Erreur : {e}")
    sys.exit(1)

# Test 5 : MediaPipe
print(f"\n{Fore.CYAN}[5/6] Test MediaPipe...{Style.RESET_ALL}")
try:
    import mediapipe as mp
    print(f"   ✅ MediaPipe version : {mp.__version__}")
except Exception as e:
    print(f"   ❌ Erreur : {e}")
    sys.exit(1)

# Test 6 : Modules du projet
print(f"\n{Fore.CYAN}[6/6] Test modules du projet...{Style.RESET_ALL}")
try:
    from config import (
        MODEL_PATH,
        BEHAVIOR_WINDOW_SIZE,
        SEQUENCE_LENGTH,
        USE_HEURISTIC,
    )
    print(f"   ✅ config.py")
    
    from detector import YOLODetector
    print(f"   ✅ detector.py")
    
    from tracker import PersonTracker
    print(f"   ✅ tracker.py")
    
    from behavior_analyzer import TrackBehaviorAnalyzer
    print(f"   ✅ behavior_analyzer.py")
    
    from behavior_classifier import BehavioralLSTMClassifier, HeuristicScorer
    print(f"   ✅ behavior_classifier.py")
    
    from utils import TrajectoryStore, FPSCounter
    print(f"   ✅ utils.py")
    
except Exception as e:
    print(f"   ❌ Erreur : {e}")
    print(f"\n{Fore.YELLOW}💡 Assurez-vous que tous les fichiers du projet sont présents{Style.RESET_ALL}")
    sys.exit(1)

# Test 7 : Test création modèle LSTM (optionnel)
print(f"\n{Fore.CYAN}[Bonus] Test création modèle LSTM...{Style.RESET_ALL}")
try:
    from behavior_classifier import BehavioralLSTMClassifier
    model = BehavioralLSTMClassifier()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Modèle créé : {num_params:,} paramètres")
    
    # Test forward pass
    dummy_input = torch.randn(1, 60, 10)  # [batch, seq, features]
    output, attention = model(dummy_input)
    print(f"   ✅ Forward pass OK : output shape {output.shape}")
    
except Exception as e:
    print(f"   ⚠️  Erreur (non critique) : {e}")

# Résumé
print(f"\n{'='*60}")
print(f"{Fore.GREEN}✅ INSTALLATION COMPLÈTE !{Style.RESET_ALL}")
print(f"{'='*60}\n")

print(f"{Fore.CYAN}📋 Prochaines étapes :{Style.RESET_ALL}")
print(f"   1. Placez vos vidéos dans videos/shoplifting et videos/normal")
print(f"   2. Pour tester rapidement : python main_behavioral.py (mode heuristique)")
print(f"   3. Pour entraîner : python train_model.py --shoplifting_dir ... --normal_dir ...")
print(f"\n{Fore.YELLOW}💡 Consultez README.md pour plus de détails{Style.RESET_ALL}\n")
