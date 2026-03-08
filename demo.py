# ============================================================
# demo.py – Démonstration simplifiée du système
# ============================================================
# Version simplifiée de main_behavioral.py pour une démo rapide
# Usage : python demo.py --video path/to/video.mp4
# ============================================================

import argparse
import cv2
import sys
from colorama import Fore, Style, init

init(autoreset=True)

from detector import YOLODetector
from tracker import PersonTracker
from behavior_analyzer import TrackBehaviorAnalyzer
from behavior_classifier import HeuristicScorer
from config import ALERT_THRESHOLD_HIGH, ALERT_THRESHOLD_MEDIUM


def run_demo(video_path, save_output=False):
    """
    Démonstration du système sur une vidéo.
    """
    print(f"\n{'='*60}")
    print(f"  🛒 DEMO - Shoplifting Detection")
    print(f"{'='*60}\n")
    
    # Initialisation
    print(f"📹 Vidéo : {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"{Fore.RED}❌ Impossible d'ouvrir la vidéo{Style.RESET_ALL}")
        sys.exit(1)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    print(f"✅ Résolution : {w}x{h}")
    print(f"✅ FPS : {fps:.0f}\n")
    
    # Modules
    print("🔧 Chargement des modules...")
    detector = YOLODetector()
    tracker = PersonTracker()
    analyzer = TrackBehaviorAnalyzer()
    scorer = HeuristicScorer()
    
    # VideoWriter optionnel
    writer = None
    if save_output:
        output_path = "demo_output.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"💾 Sauvegarde vers : {output_path}")
    
    print(f"\n{Fore.GREEN}✅ Démarrage de la détection...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Touches : Q = quitter | S = capture{Style.RESET_ALL}\n")
    
    frame_count = 0
    alerts_triggered = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Pipeline
        detections = detector.detect(frame)
        tracked_persons = tracker.update(detections, frame)
        behavior_results = analyzer.update(frame, tracked_persons)
        
        active_ids = set()
        
        for person in tracked_persons:
            if not person.confirmed:
                continue
            
            track_id = person.track_id
            active_ids.add(track_id)
            
            # Scoring
            behavior_data = behavior_results.get(track_id, {})
            score, risk_level = scorer.compute_score(track_id, behavior_data)
            score_data = scorer.get_score(track_id)
            probability = score_data['probability']
            
            # Visualisation
            x1, y1, x2, y2 = person.bbox_ltrb
            
            # Couleur selon risque
            if probability >= ALERT_THRESHOLD_HIGH:
                color = (0, 0, 255)  # Rouge
                label = f"ID {track_id} | 🚨 SUSPECT {probability*100:.0f}%"
                thickness = 4
                
                # Déclencher alerte (une fois par track)
                if track_id not in alerts_triggered:
                    print(f"\n{Fore.RED}🚨 ALERTE ! Track {track_id} | Score: {score:.1f}/100{Style.RESET_ALL}")
                    alerts_triggered.add(track_id)
                    
            elif probability >= ALERT_THRESHOLD_MEDIUM:
                color = (0, 165, 255)  # Orange
                label = f"ID {track_id} | ⚡ {probability*100:.0f}%"
                thickness = 3
            else:
                color = (0, 255, 0)  # Vert
                label = f"ID {track_id} | ✓ {probability*100:.0f}%"
                thickness = 2
            
            # Dessiner box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
            
            # Fond label
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 8),
                (x1 + text_w + 10, y1),
                color,
                -1
            )
            
            # Texte
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        
        # HUD simple
        info_text = f"Frame: {frame_count} | Personnes: {len(active_ids)} | Alertes: {len(alerts_triggered)}"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Cleanup
        scorer.cleanup(active_ids)
        
        # Sauvegarde
        if writer:
            writer.write(frame)
        
        # Affichage
        cv2.imshow("Shoplifting Detection - DEMO", frame)
        
        # Clavier
        key = cv2.waitKey(1) & 0xFF
        
        if key in (ord('q'), ord('Q')):
            break
        
        if key in (ord('s'), ord('S')):
            fname = f"demo_frame_{frame_count}.jpg"
            cv2.imwrite(fname, frame)
            print(f"{Fore.GREEN}💾 Capture : {fname}{Style.RESET_ALL}")
    
    # Nettoyage
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Résumé
    print(f"\n{'='*60}")
    print(f"{Fore.CYAN}📊 RÉSUMÉ DE LA DÉMO{Style.RESET_ALL}")
    print(f"{'='*60}")
    print(f"Frames traitées : {frame_count}")
    print(f"Alertes déclenchées : {len(alerts_triggered)}")
    
    if alerts_triggered:
        print(f"\n{Fore.RED}🚨 Tracks suspects : {sorted(alerts_triggered)}{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.GREEN}✅ Aucune activité suspecte détectée{Style.RESET_ALL}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Démo du système de détection")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Chemin vers la vidéo"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Sauvegarder la vidéo de sortie"
    )
    
    args = parser.parse_args()
    
    run_demo(args.video, args.save)


if __name__ == "__main__":
    main()
