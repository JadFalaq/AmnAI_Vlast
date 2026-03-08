# ============================================================
# visualize_features.py – Visualisation des features extraites
# ============================================================
# Utile pour comprendre ce que le modèle "voit"
# Usage : python visualize_features.py --video path/to/video.mp4
# ============================================================

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from detector import YOLODetector
from tracker import PersonTracker
from behavior_analyzer import TrackBehaviorAnalyzer




def visualize_track_features(video_path, track_id_to_plot=None):
    """
    Visualise les features comportementales d'une vidéo.
    """
    print(f"\n📊 Analyse de : {video_path}\n")
    
    # Initialisation
    detector = YOLODetector()
    tracker = PersonTracker()
    analyzer = TrackBehaviorAnalyzer()
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir {video_path}")
        return
    
    # Accumulateurs par track
    tracks_data = defaultdict(lambda: {
        'frames': [],
        'velocities': [],
        'gestures': defaultdict(list),
        'stationary': [],
    })
    
    frame_count = 0
    
    print("⏳ Traitement des frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Détection + Tracking + Analyse
        detections = detector.detect(frame)
        tracked_persons = tracker.update(detections, frame)
        behavior_results = analyzer.update(frame, tracked_persons)
        
        # Collecter les données
        for track_id, behavior_data in behavior_results.items():
            tracks_data[track_id]['frames'].append(frame_count)
            tracks_data[track_id]['velocities'].append(
                behavior_data.get('avg_velocity', 0.0)
            )
            
            gesture_counts = behavior_data.get('gesture_counts', {})
            for gesture, count in gesture_counts.items():
                tracks_data[track_id]['gestures'][gesture].append(count)
            
            tracks_data[track_id]['stationary'].append(
                behavior_data.get('stationary_ratio', 0.0)
            )
        
        if frame_count % 30 == 0:
            print(f"   Frame {frame_count} | Tracks actifs : {len(behavior_results)}")
    
    cap.release()
    
    print(f"\n✅ {frame_count} frames traitées")
    print(f"✅ {len(tracks_data)} personnes trackées\n")
    
    # Choisir track à visualiser
    if not tracks_data:
        print("❌ Aucune personne trackée dans cette vidéo")
        return
    
    if track_id_to_plot is None:
        # Prendre le track avec le plus de frames
        track_id_to_plot = max(tracks_data.keys(), key=lambda tid: len(tracks_data[tid]['frames']))
    
    if track_id_to_plot not in tracks_data:
        print(f"❌ Track ID {track_id_to_plot} non trouvé")
        return
    
    # Visualisation
    print(f"📈 Visualisation Track ID : {track_id_to_plot}")
    
    track = tracks_data[track_id_to_plot]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Features Comportementales - Track ID {track_id_to_plot}', fontsize=16)
    
    # 1. Vitesse
    axes[0].plot(track['frames'], track['velocities'], 'b-', linewidth=2)
    axes[0].set_ylabel('Vitesse (pixels/frame)', fontsize=12)
    axes[0].set_title('Mouvement', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=2.0, color='r', linestyle='--', label='Seuil lent')
    axes[0].axhline(y=15.0, color='orange', linestyle='--', label='Seuil rapide')
    axes[0].legend()
    
    # 2. Ratio stationnaire
    axes[1].plot(track['frames'], track['stationary'], 'g-', linewidth=2)
    axes[1].set_ylabel('Ratio Stationnaire', fontsize=12)
    axes[1].set_title('Temps Immobile', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Seuil suspect')
    axes[1].legend()
    axes[1].set_ylim([0, 1])
    
    # 3. Gestes suspects
    gestures = track['gestures']
    if gestures:
        for gesture_name, counts in gestures.items():
            if counts:
                axes[2].plot(track['frames'][:len(counts)], counts, marker='o', label=gesture_name)
    
    axes[2].set_xlabel('Frame', fontsize=12)
    axes[2].set_ylabel('Nombre cumulatif', fontsize=12)
    axes[2].set_title('Gestes Suspects Détectés', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    output_path = f"features_track_{track_id_to_plot}.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n💾 Graphique sauvegardé : {output_path}")
    
    plt.show()
    
    # Statistiques
    print(f"\n📊 Statistiques Track ID {track_id_to_plot} :")
    print(f"   Durée : {len(track['frames'])} frames")
    print(f"   Vitesse moyenne : {np.mean(track['velocities']):.2f} px/frame")
    print(f"   Vitesse max : {np.max(track['velocities']):.2f} px/frame")
    print(f"   Ratio stationnaire moyen : {np.mean(track['stationary']):.2%}")
    
    if gestures:
        print(f"\n   Gestes suspects :")
        for gesture, counts in gestures.items():
            if counts:
                print(f"      {gesture} : {counts[-1]} fois")


def main():
    parser = argparse.ArgumentParser(description="Visualiser les features comportementales")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Chemin vers la vidéo à analyser"
    )
    parser.add_argument(
        "--track_id",
        type=int,
        default=None,
        help="ID du track à visualiser (défaut: track le plus long)"
    )

    args = parser.parse_args()

    visualize_track_features(args.video, args.track_id)


if __name__ == "__main__":
    main()
