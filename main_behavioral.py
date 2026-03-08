# ============================================================
# main.py – Détection + Tracking + Analyse comportementale
# ============================================================
# NOUVEAU : Intègre la détection d'activités suspectes
#
# COMMENT LANCER :
#   python main.py
#
# Pour mode heuristique (sans entraînement) :
#   config.py → USE_HEURISTIC = True
#
# Pour mode LSTM (après entraînement) :
#   config.py → USE_LSTM = True
#   config.py → MODEL_PATH_LSTM = "model_shoplifting.pth"
# ============================================================
import os
import sys
import cv2
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

# --- Nos modules ---
from config import (
    VIDEO_SOURCE,
    SAVE_OUTPUT_VIDEO,
    OUTPUT_VIDEO_PATH,
    USE_HEURISTIC,
    USE_LSTM,
    MODEL_PATH_LSTM,
    ALERT_THRESHOLD_HIGH,
    ALERT_THRESHOLD_MEDIUM,
    COLOR_HIGH_RISK,
    COLOR_MEDIUM_RISK,
    COLOR_NORMAL,
)

from detector import YOLODetector
from tracker import PersonTracker
from behavior_analyzer import TrackBehaviorAnalyzer
from behavior_classifier import ShopliftingDetector, HeuristicScorer
from utils import (
    TrajectoryStore,
    draw_all,
    draw_hud,
    print_tracking_log,
    FPSCounter,
)


# ============================================================
# 🎨 Nouvelles fonctions de visualisation
# ============================================================

def draw_risk_box(frame, bbox_ltrb, track_id, probability, risk_level):
    """
    Dessine un box avec couleur selon le niveau de risque.
    """
    x1, y1, x2, y2 = bbox_ltrb
    
    # Couleur selon risque
    if risk_level == "HIGH_RISK":
        color = COLOR_HIGH_RISK
        thickness = 4
        label = f"ID {track_id} | ⚠️ SUSPECT {probability*100:.0f}%"
    elif risk_level == "MEDIUM_RISK":
        color = COLOR_MEDIUM_RISK
        thickness = 3
        label = f"ID {track_id} | ⚡ {probability*100:.0f}%"
    else:
        color = COLOR_NORMAL
        thickness = 2
        label = f"ID {track_id} | ✓ {probability*100:.0f}%"
    
    # Rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
    
    # Position du label
    if y1 - text_h - 8 < 0:
        label_bg_y1 = y1
        label_bg_y2 = y1 + text_h + 8
        text_y = y1 + text_h + 2
    else:
        label_bg_y1 = y1 - text_h - 8
        label_bg_y2 = y1
        text_y = y1 - 4
    
    # Fond du label
    cv2.rectangle(
        frame,
        (x1, label_bg_y1),
        (x1 + text_w + 10, label_bg_y2),
        color,
        -1
    )
    
    # Texte
    cv2.putText(
        frame,
        label,
        (x1 + 5, text_y),
        font,
        font_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def draw_behavior_hud(frame, fps, num_confirmed, num_high_risk, num_medium_risk):
    """
    HUD amélioré avec statistiques de risque.
    """
    lines = [
        f"FPS: {fps:.1f}",
        f"Personnes: {num_confirmed}",
        f"🚨 Risque élevé: {num_high_risk}",
        f"⚡ Suspect: {num_medium_risk}",
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    line_height = 24
    padding = 8
    
    max_width = max(cv2.getTextSize(l, font, font_scale, 1)[0][0] for l in lines)
    bg_w = max_width + padding * 2
    bg_h = len(lines) * line_height + padding
    
    # Fond semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + bg_w, 5 + bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Texte
    for i, line in enumerate(lines):
        # Couleur selon ligne
        if "Risque élevé" in line and num_high_risk > 0:
            color = COLOR_HIGH_RISK
        elif "Suspect" in line and num_medium_risk > 0:
            color = COLOR_MEDIUM_RISK
        elif "FPS" in line and fps < 15:
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)
        
        cv2.putText(
            frame,
            line,
            (5 + padding, 5 + padding + (i + 1) * line_height - 4),
            font,
            font_scale,
            color,
            1,
            cv2.LINE_AA
        )


def trigger_alert(track_id, probability, risk_level):
    """
    Déclenche une alerte visuelle/sonore.
    """
    if risk_level == "HIGH_RISK":
        print(f"\n{Fore.RED}{'='*60}")
        print(f"🚨 ALERTE CRITIQUE ! Track ID {track_id}")
        print(f"   Probabilité shoplifting : {probability*100:.1f}%")
        print(f"   Action recommandée : Intervention immédiate")
        print(f"{'='*60}{Style.RESET_ALL}\n")
    elif risk_level == "MEDIUM_RISK":
        print(f"\n{Fore.YELLOW}⚠️  SUSPECT - Track ID {track_id} ({probability*100:.1f}%){Style.RESET_ALL}")


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
    print(f"\n{'=' * 60}")
    print(f"  🛒 SHOPLIFTING DETECTION - Détection comportementale")
    print(f"{'=' * 60}\n")

    # --- Charger YOLO + DeepSORT ---
    detector = YOLODetector()
    tracker = PersonTracker()
    
    # --- Analyse comportementale ---
    behavior_analyzer = TrackBehaviorAnalyzer()
    
    # --- Classificateur ---
    if USE_LSTM and MODEL_PATH_LSTM:
        print(f"{Fore.CYAN}[CLASSIFIER] Mode LSTM activé{Style.RESET_ALL}")
        shoplifting_detector = ShopliftingDetector(
            model_path=MODEL_PATH_LSTM,
            device="cuda"
        )
        use_ml = True
    elif USE_HEURISTIC:
        print(f"{Fore.CYAN}[CLASSIFIER] Mode HEURISTIQUE activé (baseline){Style.RESET_ALL}")
        heuristic_scorer = HeuristicScorer()
        use_ml = False
    else:
        print(f"{Fore.RED}[CLASSIFIER] ❌ Aucun mode activé ! Activez USE_HEURISTIC ou USE_LSTM{Style.RESET_ALL}")
        sys.exit(1)

    # --- Caméra ---
    cap = init_camera(VIDEO_SOURCE)
    writer = init_video_writer(cap)

    # --- Utilitaires ---
    fps_counter = FPSCounter()
    traj_store = TrajectoryStore(max_history=90)

    frame_count = 0
    print_every = 30

    print(f"\n{Fore.GREEN}[MAIN] ✅ Système prêt ! Détection en cours...{Style.RESET_ALL}")
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
        # --------------------------------------------------------
        detections = detector.detect(frame)

        # --------------------------------------------------------
        # ÉTAPE 2 : DeepSORT tracking
        # --------------------------------------------------------
        tracked_persons = tracker.update(detections, frame)

        # --------------------------------------------------------
        # ÉTAPE 3 : Analyse comportementale
        # --------------------------------------------------------
        behavior_results = behavior_analyzer.update(frame, tracked_persons)

        # --------------------------------------------------------
        # ÉTAPE 4 : Classification (LSTM ou Heuristique)
        # --------------------------------------------------------
        active_ids = set()
        num_high_risk = 0
        num_medium_risk = 0
        
        for person in tracked_persons:
            if not person.confirmed:
                continue
            
            track_id = person.track_id
            active_ids.add(track_id)
            
            # Récupérer behavior data
            behavior_data = behavior_results.get(track_id, {})
            
            # Classification
            if use_ml:
                has_prediction, probability = shoplifting_detector.predict(
                    track_id, behavior_data
                )
                
                if has_prediction:
                    pred_data = shoplifting_detector.get_prediction(track_id)
                    probability = pred_data['probability']
                    is_suspicious = pred_data['is_suspicious']
                else:
                    probability = 0.0
                    is_suspicious = False
            else:
                score, risk_level = heuristic_scorer.compute_score(
                    track_id, behavior_data
                )
                score_data = heuristic_scorer.get_score(track_id)
                probability = score_data['probability']
                is_suspicious = score_data['risk_level'] != 'NORMAL'
            
            # Déterminer risk level
            if probability >= ALERT_THRESHOLD_HIGH:
                risk_level = "HIGH_RISK"
                num_high_risk += 1
                trigger_alert(track_id, probability, risk_level)
            elif probability >= ALERT_THRESHOLD_MEDIUM:
                risk_level = "MEDIUM_RISK"
                num_medium_risk += 1
                trigger_alert(track_id, probability, risk_level)
            else:
                risk_level = "NORMAL"
            
            # --------------------------------------------------------
            # ÉTAPE 5 : Dessiner avec couleur selon risque
            # --------------------------------------------------------
            draw_risk_box(
                frame,
                person.bbox_ltrb,
                track_id,
                probability,
                risk_level
            )
            
            # Trajectoire
            traj_store.update(track_id, person.bbox_ltrb)

        # Cleanup
        if use_ml:
            shoplifting_detector.cleanup(active_ids)
        else:
            heuristic_scorer.cleanup(active_ids)

        # --------------------------------------------------------
        # ÉTAPE 6 : HUD amélioré
        # --------------------------------------------------------
        fps_counter.tick()
        num_confirmed = sum(1 for p in tracked_persons if p.confirmed)
        
        draw_behavior_hud(
            frame,
            fps_counter.get_fps(),
            num_confirmed,
            num_high_risk,
            num_medium_risk
        )

        # --------------------------------------------------------
        # Log terminal
        # --------------------------------------------------------
        if frame_count % print_every == 0:
            print(f"{Fore.CYAN}[Frame {frame_count}] "
                  f"Personnes: {num_confirmed} | "
                  f"🚨 Risque élevé: {num_high_risk} | "
                  f"⚡ Suspects: {num_medium_risk}{Style.RESET_ALL}")

        # --------------------------------------------------------
        # Sauvegarde vidéo
        # --------------------------------------------------------
        if writer:
            writer.write(frame)

        # --------------------------------------------------------
        # Affichage fenêtre
        # --------------------------------------------------------
        cv2.imshow("Shoplifting Detection - Behavioral Analysis", frame)

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

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"  ✅ Terminé – {frame_count} frames traitées")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")


# ============================================================
if __name__ == "__main__":
    main()