# ============================================================
# utils.py — Visualisation du tracking
# ============================================================
# Dessine sur chaque frame :
#   - Bounding box par personne (vert confirmé / gris pending)
#   - Label "ID : X" au-dessus du box
#   - Ligne de trajectoire (historique des positions)
#   - HUD : FPS + nombre de personnes
# ============================================================

import cv2
import time
import numpy as np
from collections import defaultdict
from colorama import Fore, Style

from config import (
    COLOR_CONFIRMED,
    COLOR_UNCONFIRMED,
    COLOR_FPS,
)


# ============================================================
# 📦 Classe qui garde l'historique des positions par ID
#     → Pour dessiner la trajectoire de chaque personne
# ============================================================
class TrajectoryStore:
    """
    Stocke les dernières positions (centres) de chaque personne
    pour dessiner la ligne de trajectoire.
    
    max_history : nombre de points gardés par personne.
        À 30fps, 90 points = 3 secondes de trajectoire visible.
    """
    def __init__(self, max_history: int = 90):
        self.max_history = max_history
        # dict : { track_id → liste de (cx, cy) }
        self.history = defaultdict(list)

    def update(self, track_id: int, bbox_ltrb: list):
        """Ajoute le centre actuel de la personne à l'historique."""
        x1, y1, x2, y2 = bbox_ltrb
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        self.history[track_id].append((cx, cy))

        # Garder uniquement les derniers points
        if len(self.history[track_id]) > self.max_history:
            self.history[track_id].pop(0)

    def get_trajectory(self, track_id: int) -> list:
        """Retourne la liste des points (cx, cy) pour cet ID."""
        return self.history.get(track_id, [])

    def cleanup(self, active_ids: set):
        """Supprime les historiques des IDs qui ne sont plus actifs."""
        dead_ids = set(self.history.keys()) - active_ids
        for did in dead_ids:
            del self.history[did]


# ============================================================
# 📐 Dessiner un box + label "ID : X"
# ============================================================
def draw_person_box(frame: np.ndarray, bbox_ltrb: list, track_id: int, confirmed: bool):
    """
    Dessine le rectangle + le label ID sur la frame.
    
    Args:
        frame     : Image numpy (modifiée en place)
        bbox_ltrb : [x1, y1, x2, y2]
        track_id  : ID unique de la personne
        confirmed : True = vert, False = gris
    """
    x1, y1, x2, y2 = bbox_ltrb
    color = COLOR_CONFIRMED if confirmed else COLOR_UNCONFIRMED
    thickness = 3 if confirmed else 2

    # --- Rectangle ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # --- Label "ID : X" ---
    label = f"ID : {track_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)

    # Position du fond du label : au-dessus du box
    # Si ça dépasse en haut → on le met en dessous
    if y1 - text_h - 8 < 0:
        label_bg_y1 = y1
        label_bg_y2 = y1 + text_h + 8
        text_y = y1 + text_h + 2
    else:
        label_bg_y1 = y1 - text_h - 8
        label_bg_y2 = y1
        text_y = y1 - 4

    # Fond du label
    cv2.rectangle(frame, (x1, label_bg_y1), (x1 + text_w + 10, label_bg_y2), color, -1)

    # Texte blanc dessus
    cv2.putText(
        frame, label,
        (x1 + 5, text_y),
        font, font_scale,
        (255, 255, 255),
        2, cv2.LINE_AA
    )


# ============================================================
# 📈 Dessiner la ligne de trajectoire
# ============================================================
def draw_trajectory(frame: np.ndarray, points: list, color: tuple):
    """
    Dessine une ligne entre tous les points de la trajectoire.
    La ligne devient plus claire au plus elle est récente.
    
    Args:
        frame  : Image numpy
        points : Liste de (cx, cy)
        color  : Couleur de base (BGR)
    """
    if len(points) < 2:
        return

    num_points = len(points)

    for i in range(1, num_points):
        # Transparence progressive : plus ancien = plus sombre
        alpha = i / num_points  # 0.0 (ancien) → 1.0 (récent)
        line_color = (
            int(color[0] * alpha),
            int(color[1] * alpha),
            int(color[2] * alpha),
        )
        thickness = max(1, int(3 * alpha))  # Plus ancien = plus fin

        cv2.line(
            frame,
            points[i - 1],
            points[i],
            line_color,
            thickness,
            cv2.LINE_AA,
        )


# ============================================================
# 🎯 Dessiner toutes les personnes + trajectoires sur la frame
# ============================================================
def draw_all(frame: np.ndarray, tracked_persons: list, traj_store: TrajectoryStore):
    """
    Orchestre tout le dessin sur une frame.
    
    Args:
        frame           : Image numpy
        tracked_persons : Liste de TrackedPerson
        traj_store      : TrajectoryStore pour les lignes de trajectoire
    """
    active_ids = set()

    for person in tracked_persons:
        tid = person.track_id
        active_ids.add(tid)

        # --- Mettre à jour l'historique de position ---
        traj_store.update(tid, person.bbox_ltrb)

        # --- Dessiner la trajectoire ---
        color = COLOR_CONFIRMED if person.confirmed else COLOR_UNCONFIRMED
        trajectory_points = traj_store.get_trajectory(tid)
        draw_trajectory(frame, trajectory_points, color)

        # --- Dessiner le box + label ID ---
        draw_person_box(frame, person.bbox_ltrb, tid, person.confirmed)

    # --- Nettoyer les anciens historiques ---
    traj_store.cleanup(active_ids)


# ============================================================
# ⏱️  Compteur FPS
# ============================================================
class FPSCounter:
    def __init__(self, smoothing: int = 15):
        self.timestamps = []
        self.smoothing = smoothing

    def tick(self):
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.smoothing + 1:
            self.timestamps.pop(0)

    def get_fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ============================================================
# 📊 HUD : panneau info en haut à gauche
# ============================================================
def draw_hud(frame: np.ndarray, fps: float, num_confirmed: int, num_total: int):
    """
    Affiche :
        - FPS (rouge si < 15)
        - Personnes confirmées
        - Personnes en attente de confirmation
    """
    lines = [
        f"FPS: {fps:.1f}",
        f"Personnes: {num_confirmed}",
        f"En attente: {num_total - num_confirmed}",
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
        color = COLOR_FPS
        if i == 0 and fps < 15:
            color = (0, 0, 255)  # Rouge si FPS bas

        cv2.putText(
            frame, line,
            (5 + padding, 5 + padding + (i + 1) * line_height - 4),
            font, font_scale,
            color, 1, cv2.LINE_AA,
        )


# ============================================================
# 🖨️  Log terminal
# ============================================================
def print_tracking_log(tracked_persons: list, frame_num: int):
    """Affiche un résumé dans le terminal toutes les N frames."""
    print(f"\n{Fore.CYAN}--- Frame #{frame_num} ---{Style.RESET_ALL}")

    if not tracked_persons:
        print(f"  {Fore.RED}Aucune personne détectée{Style.RESET_ALL}")
        return

    for p in tracked_persons:
        status = f"{Fore.GREEN}✅ confirmée{Style.RESET_ALL}" if p.confirmed else f"{Fore.YELLOW}⏳ en attente{Style.RESET_ALL}"
        print(f"  👤 ID={p.track_id:>3d}  |  {status}  |  bbox={p.bbox_ltrb}")