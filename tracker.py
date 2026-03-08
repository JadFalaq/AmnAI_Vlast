# ============================================================
# tracker.py — Wrapper DeepSORT pour le tracking de personnes
# ============================================================
# Responsabilités :
#   1. Initialiser DeepSORT avec nos paramètres
#   2. Passer les détections YOLO → DeepSORT chaque frame
#   3. Retourner les tracks avec leur ID unique + bbox
# ============================================================

from deep_sort_realtime.deepsort_tracker import DeepSort
from colorama import Fore, Style

from config import (
    DEEPSORT_MAX_AGE,
    DEEPSORT_N_INIT,
    DEEPSORT_MAX_COSINE_DIST,
)


class TrackedPerson:
    """
    Représente une personne trackée avec son ID unique.
    
    Attributs :
        track_id   : int    — ID unique assigné par DeepSORT
        bbox_ltrb  : list   — [left, top, right, bottom] = [x1, y1, x2, y2]
        confirmed  : bool   — True si le track est confirmé (après n_init frames)
    """
    def __init__(self, track_id: int, bbox_ltrb: list, confirmed: bool):
        self.track_id = track_id
        self.bbox_ltrb = bbox_ltrb      # [x1, y1, x2, y2]
        self.confirmed = confirmed

    def __repr__(self):
        status = "✅ confirmed" if self.confirmed else "⏳ pending"
        return f"TrackedPerson(ID={self.track_id}, {status}, bbox={self.bbox_ltrb})"


class PersonTracker:
    """
    Classe principale du tracking.
    
    Utilisation :
        tracker = PersonTracker()
        
        # Chaque frame :
        detections = detector.detect(frame)          # liste YOLO
        tracked = tracker.update(detections, frame)  # liste TrackedPerson
        
        for person in tracked:
            print(person.track_id, person.bbox_ltrb)
    """

    def __init__(self):
        print(f"{Fore.CYAN}[TRACKER] Initialisation DeepSORT...{Style.RESET_ALL}")

        self.tracker = DeepSort(
            max_age=DEEPSORT_MAX_AGE,               # frames avant suppression du track
            n_init=DEEPSORT_N_INIT,                  # frames avant confirmation
            max_cosine_distance=DEEPSORT_MAX_COSINE_DIST,  # seuil similarité apparence
            embedder="mobilenet",                    # embedder intégré, léger pour GTX 1650
            half=True,                               # FP16 → plus rapide sur GPU
            bgr=True,                                # nos frames sont en BGR (OpenCV)
            embedder_gpu=True,                       # embedder sur GPU aussi
        )

        print(f"{Fore.GREEN}[TRACKER] ✅ DeepSORT initialisé{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[TRACKER]    max_age={DEEPSORT_MAX_AGE} | n_init={DEEPSORT_N_INIT} | cosine={DEEPSORT_MAX_COSINE_DIST}{Style.RESET_ALL}")

    def update(self, detections: list, frame) -> list:
        """
        Met à jour le tracker avec les nouvelles détections.
        
        Args:
            detections : Liste de tuples YOLO →
                         [ ([left,top,w,h], confidence, "person"), ... ]
            frame      : La frame actuelle (numpy BGR) pour l'embedder
                         (DeepSORT extrait une crop de chaque personne
                          pour comparer les apparences)
        
        Returns:
            Liste de TrackedPerson (confirmés + non confirmés)
        """
        # --- Passer les détections à DeepSORT ---
        # DeepSORT retourne une liste de Track objects
        tracks = self.tracker.update_tracks(detections, frame=frame)

        tracked_persons = []

        for track in tracks:
            # --- track.is_confirmed() : True après n_init détections ---
            # On garde aussi les non confirmés pour les afficher en gris
            
            track_id = int(track.track_id)  # DeepSORT retourne un str, on caste en int
            bbox_ltrb = track.to_ltrb()  # Retourne [left, top, right, bottom]
            confirmed = track.is_confirmed()

            tracked_persons.append(
                TrackedPerson(
                    track_id=track_id,
                    bbox_ltrb=[int(v) for v in bbox_ltrb],
                    confirmed=confirmed,
                )
            )

        return tracked_persons