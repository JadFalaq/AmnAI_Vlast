# ============================================================
# behavior_analyzer.py – Extraction de features comportementales
# ============================================================
# Responsabilités :
#   1. Extraire la pose avec MediaPipe (léger, optimisé GPU)
#   2. Calculer les features de comportement par track
#   3. Détecter les gestes suspects (main→poche, penché, etc.)
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

from config import BEHAVIOR_WINDOW_SIZE, STATIONARY_THRESHOLD


class PoseExtractor:
    """
    Extrait la pose humaine avec MediaPipe Pose.
    Optimisé pour GTX 1650 (4GB VRAM).
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0 = Lite (plus rapide, parfait pour GTX 1650)
            smooth_landmarks=True,
            enable_segmentation=False,  # Désactiver pour économiser VRAM
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_landmarks(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extrait les landmarks de pose pour une personne dans sa bbox.
        
        Args:
            frame: Frame complète BGR
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Array numpy (33, 3) avec [x, y, visibility] ou None si échec
        """
        x1, y1, x2, y2 = bbox
        
        # Crop avec marge de sécurité
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None
        
        # MediaPipe attend RGB
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(person_rgb)
        
        if results.pose_landmarks is None:
            return None
        
        # Convertir en numpy array
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.visibility])
        
        return np.array(landmarks)
    
    def __del__(self):
        self.pose.close()


class BehaviorFeatures:
    """
    Calcule les features comportementales à partir de la pose et du mouvement.
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose.PoseLandmark
        
    def detect_hand_to_pocket(self, landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte si la main va vers la poche/taille.
        
        Returns:
            (is_suspicious, confidence)
        """
        if landmarks is None:
            return False, 0.0
        
        # Landmarks : poignet droit/gauche et hanches
        right_wrist = landmarks[self.mp_pose.RIGHT_WRIST.value]
        left_wrist = landmarks[self.mp_pose.LEFT_WRIST.value]
        right_hip = landmarks[self.mp_pose.RIGHT_HIP.value]
        left_hip = landmarks[self.mp_pose.LEFT_HIP.value]
        
        # Distance main droite → hanche droite
        dist_right = np.sqrt(
            (right_wrist[0] - right_hip[0])**2 + 
            (right_wrist[1] - right_hip[1])**2
        )
        
        # Distance main gauche → hanche gauche
        dist_left = np.sqrt(
            (left_wrist[0] - left_hip[0])**2 + 
            (left_wrist[1] - left_hip[1])**2
        )
        
        # Seuil : si main proche de la hanche (< 0.15 en coordonnées normalisées)
        threshold = 0.15
        
        if dist_right < threshold or dist_left < threshold:
            confidence = max(
                (threshold - dist_right) / threshold,
                (threshold - dist_left) / threshold
            )
            return True, min(confidence, 1.0)
        
        return False, 0.0
    
    def detect_hand_to_bag(self, landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte si la main va vers l'épaule (sac à dos).
        """
        if landmarks is None:
            return False, 0.0
        
        right_wrist = landmarks[self.mp_pose.RIGHT_WRIST.value]
        left_wrist = landmarks[self.mp_pose.LEFT_WRIST.value]
        right_shoulder = landmarks[self.mp_pose.RIGHT_SHOULDER.value]
        left_shoulder = landmarks[self.mp_pose.LEFT_SHOULDER.value]
        
        # Distance main → épaule opposée (geste de mettre dans sac à dos)
        dist_right_to_left_shoulder = np.sqrt(
            (right_wrist[0] - left_shoulder[0])**2 + 
            (right_wrist[1] - left_shoulder[1])**2
        )
        
        dist_left_to_right_shoulder = np.sqrt(
            (left_wrist[0] - right_shoulder[0])**2 + 
            (left_wrist[1] - right_shoulder[1])**2
        )
        
        threshold = 0.2
        
        if dist_right_to_left_shoulder < threshold or dist_left_to_right_shoulder < threshold:
            confidence = max(
                (threshold - dist_right_to_left_shoulder) / threshold,
                (threshold - dist_left_to_right_shoulder) / threshold
            )
            return True, min(confidence, 1.0)
        
        return False, 0.0
    
    def detect_body_bent(self, landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte si le corps est penché (dissimulation).
        """
        if landmarks is None:
            return False, 0.0
        
        # Angle du torse : épaule → hanche
        right_shoulder = landmarks[self.mp_pose.RIGHT_SHOULDER.value]
        right_hip = landmarks[self.mp_pose.RIGHT_HIP.value]
        
        # Calcul de l'angle par rapport à la verticale
        dy = right_hip[1] - right_shoulder[1]
        dx = right_hip[0] - right_shoulder[0]
        
        angle = np.abs(np.arctan2(dx, dy) * 180 / np.pi)
        
        # Si l'angle > 25° → corps penché
        threshold = 25.0
        
        if angle > threshold:
            confidence = min((angle - threshold) / 30.0, 1.0)
            return True, confidence
        
        return False, 0.0
    
    def detect_looking_around(self, landmarks: np.ndarray, prev_landmarks: Optional[np.ndarray]) -> Tuple[bool, float]:
        """
        Détecte les mouvements rapides de tête (regard furtif).
        """
        if landmarks is None or prev_landmarks is None:
            return False, 0.0
        
        nose_curr = landmarks[self.mp_pose.NOSE.value]
        nose_prev = prev_landmarks[self.mp_pose.NOSE.value]
        
        # Mouvement du nez (indicateur de rotation tête)
        movement = np.sqrt(
            (nose_curr[0] - nose_prev[0])**2 + 
            (nose_curr[1] - nose_prev[1])**2
        )
        
        # Si mouvement > seuil → regard furtif
        threshold = 0.05
        
        if movement > threshold:
            confidence = min(movement / (threshold * 3), 1.0)
            return True, confidence
        
        return False, 0.0


class TrackBehaviorAnalyzer:
    """
    Analyse le comportement d'une personne sur une séquence de frames.
    Garde l'historique pour chaque track_id.
    """
    
    def __init__(self):
        self.pose_extractor = PoseExtractor()
        self.behavior_features = BehaviorFeatures()
        
        # Historique par track_id
        self.tracks_history = defaultdict(lambda: {
            'landmarks_history': deque(maxlen=BEHAVIOR_WINDOW_SIZE),
            'positions_history': deque(maxlen=BEHAVIOR_WINDOW_SIZE),
            'velocities': deque(maxlen=BEHAVIOR_WINDOW_SIZE),
            'gesture_counts': defaultdict(int),
            'frames_count': 0,
            'time_stationary': 0,
            'last_position': None,
        })
    
    def update(self, frame: np.ndarray, tracked_persons: List) -> Dict[int, Dict]:
        """
        Met à jour l'analyse comportementale pour toutes les personnes trackées.
        
        Args:
            frame: Frame actuelle BGR
            tracked_persons: Liste de TrackedPerson
            
        Returns:
            Dict {track_id: behavior_data}
        """
        behavior_results = {}
        active_ids = set()
        
        for person in tracked_persons:
            if not person.confirmed:
                continue
            
            track_id = person.track_id
            active_ids.add(track_id)
            
            bbox = person.bbox_ltrb
            
            # Extraire pose
            landmarks = self.pose_extractor.extract_landmarks(frame, bbox)
            
            track_data = self.tracks_history[track_id]
            track_data['frames_count'] += 1
            
            # Calculer position (centre bbox)
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            track_data['positions_history'].append(center)
            
            # Calculer vitesse
            velocity = 0.0
            if track_data['last_position'] is not None:
                prev_x, prev_y = track_data['last_position']
                curr_x, curr_y = center
                velocity = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            
            track_data['velocities'].append(velocity)
            track_data['last_position'] = center
            
            # Temps stationnaire
            if velocity < STATIONARY_THRESHOLD:
                track_data['time_stationary'] += 1
            
            # Analyse des gestes si landmarks disponibles
            if landmarks is not None:
                track_data['landmarks_history'].append(landmarks)
                
                # Détection gestes
                hand_pocket, conf_pocket = self.behavior_features.detect_hand_to_pocket(landmarks)
                hand_bag, conf_bag = self.behavior_features.detect_hand_to_bag(landmarks)
                body_bent, conf_bent = self.behavior_features.detect_body_bent(landmarks)
                
                if hand_pocket:
                    track_data['gesture_counts']['hand_to_pocket'] += 1
                if hand_bag:
                    track_data['gesture_counts']['hand_to_bag'] += 1
                if body_bent:
                    track_data['gesture_counts']['body_bent'] += 1
                
                # Regard furtif (comparaison avec frame précédente)
                if len(track_data['landmarks_history']) >= 2:
                    prev_landmarks = track_data['landmarks_history'][-2]
                    looking, conf_look = self.behavior_features.detect_looking_around(
                        landmarks, prev_landmarks
                    )
                    if looking:
                        track_data['gesture_counts']['looking_around'] += 1
            
            # Compiler les résultats
            behavior_results[track_id] = self._compile_behavior_data(track_data)
        
        # Nettoyer les tracks inactifs
        self._cleanup_inactive_tracks(active_ids)
        
        return behavior_results
    
    def _compile_behavior_data(self, track_data: Dict) -> Dict:
        """
        Compile toutes les features comportementales d'un track.
        """
        velocities = list(track_data['velocities'])
        
        return {
            'frames_count': track_data['frames_count'],
            'avg_velocity': np.mean(velocities) if velocities else 0.0,
            'velocity_std': np.std(velocities) if velocities else 0.0,
            'max_velocity': np.max(velocities) if velocities else 0.0,
            'time_stationary': track_data['time_stationary'],
            'stationary_ratio': track_data['time_stationary'] / max(track_data['frames_count'], 1),
            'gesture_counts': dict(track_data['gesture_counts']),
            'total_gestures': sum(track_data['gesture_counts'].values()),
        }
    
    def _cleanup_inactive_tracks(self, active_ids: set):
        """
        Supprime l'historique des tracks qui ne sont plus actifs.
        """
        dead_ids = set(self.tracks_history.keys()) - active_ids
        for track_id in dead_ids:
            del self.tracks_history[track_id]
