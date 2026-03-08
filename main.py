# ============================================================
# main.py — Point d'entrée : Détection + Tracking temps réel
# ============================================================
#
# COMMENT LANCER :
#   python main.py
#
# AVANT (si pas encore fait) :
#   1) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   2) pip install -r requirements.txt
#
# ============================================================

import sys
import cv2
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

# --- Nos modules ---
from config import VIDEO_SOURCE, SAVE_OUTPUT_VIDEO, OUTPUT_VIDEO_PATH
from detector import YOLODetector
from tracker import PersonTracker
from utils import (
    TrajectoryStore,
    draw_all,
    draw_hud,
    print_tracking_log,
    FPSCounter,
)


# ============================================================
# 📷 Ouvrir la source vidéo
# ============================================================
def init_camera(source) -> cv2.VideoCapture:
    print(f"{Fore.CYAN}[CAMÉRA] Ouverture de la source : {source}{Style.RESET_ALL}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"{Fore.RED}[CAMÉRA] ❌ Impossible d'ouvrir la source vidéo !{Style.RESET_ALL}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{Fore.GREEN}[CAMÉRA] ✅ {w}x{h} | {fps:.0f} fps{Style.RESET_ALL}")
    return cap


# ============================================================
# 💾 VideoWriter (optionnel)
# ============================================================
def init_video_writer(cap: cv2.VideoCapture):
    if not SAVE_OUTPUT_VIDEO:
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
    print(f"{Fore.CYAN}[VIDÉO] 💾 Sauvegarde → {OUTPUT_VIDEO_PATH}{Style.RESET_ALL}")
    return writer


# ============================================================
# 🔄 BOUCLE PRINCIPALE
# ============================================================
def main():
    print(f"\n{'=' * 55}")
    print(f"  🏪  SHOPLIFTING DETECTION — Détection + Tracking")
    print(f"{'=' * 55}\n")

    # --- Charger YOLO + DeepSORT ---
    detector = YOLODetector()
    tracker  = PersonTracker()

    # --- Caméra ---
    cap    = init_camera(VIDEO_SOURCE)
    writer = init_video_writer(cap)

    # --- Utilitaires ---
    fps_counter = FPSCounter()
    traj_store  = TrajectoryStore(max_history=90)  # 3 sec à 30fps

    frame_count  = 0
    print_every  = 30  # log terminal toutes les 30 frames

    print(f"\n{Fore.GREEN}[MAIN] ✅ Tout prêt ! Détection + Tracking en cours...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[MAIN] Touches : Q = quitter | S = sauvegarder frame{Style.RESET_ALL}\n")

    # ============================================================
    # BOUCLE FRAME PAR FRAME
    # ============================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\n{Fore.YELLOW}[MAIN] Fin de la vidéo.{Style.RESET_ALL}")
            break

        frame_count += 1

        # --------------------------------------------------------
        # ÉTAPE 1 : YOLO détecte les personnes
        #   → retourne [ ([l,t,w,h], conf, "person"), ... ]
        # --------------------------------------------------------
        detections = detector.detect(frame)

        # --------------------------------------------------------
        # ÉTAPE 2 : DeepSORT assigne un ID unique à chaque personne
        #   → retourne [ TrackedPerson(id, bbox, confirmed), ... ]
        # --------------------------------------------------------
        tracked_persons = tracker.update(detections, frame)

        # --------------------------------------------------------
        # ÉTAPE 3 : Dessiner sur la frame
        #   → boxes, labels ID, trajectoires
        # --------------------------------------------------------
        draw_all(frame, tracked_persons, traj_store)

        # --------------------------------------------------------
        # ÉTAPE 4 : HUD (FPS + compteurs)
        # --------------------------------------------------------
        fps_counter.tick()
        num_confirmed = sum(1 for p in tracked_persons if p.confirmed)
        draw_hud(frame, fps_counter.get_fps(), num_confirmed, len(tracked_persons))

        # --------------------------------------------------------
        # Log terminal (pas à chaque frame pour ne pas spammer)
        # --------------------------------------------------------
        if frame_count % print_every == 0:
            print_tracking_log(tracked_persons, frame_count)

        # --------------------------------------------------------
        # Sauvegarde vidéo
        # --------------------------------------------------------
        if writer:
            writer.write(frame)

        # --------------------------------------------------------
        # Affichage fenêtre
        # --------------------------------------------------------
        cv2.imshow("Shoplifting Detection — Tracking", frame)

        # --------------------------------------------------------
        # Clavier
        # --------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            print(f"\n{Fore.YELLOW}[MAIN] Fermeture.{Style.RESET_ALL}")
            break

        if key in (ord("s"), ord("S")):
            fname = f"frame_{frame_count}.jpg"
            cv2.imwrite(fname, frame)
            print(f"{Fore.GREEN}[MAIN] 💾 Frame sauvegardée : {fname}{Style.RESET_ALL}")

    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n{Fore.CYAN}{'=' * 55}")
    print(f"  ✅ Terminé — {frame_count} frames traitées")
    print(f"{'=' * 55}{Style.RESET_ALL}\n")


# ============================================================
if __name__ == "__main__":
    main()