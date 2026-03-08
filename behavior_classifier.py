# ============================================================
# behavior_classifier.py – Classificateur CNN-LSTM comportemental
# ============================================================
# Architecture :
#   - Input : Séquence de features comportementales (60 frames = 2 sec)
#   - LSTM bidirectionnel pour patterns temporels
#   - Attention pour focus sur moments critiques
#   - Output : Probabilité shoplifting (0-1)
# ============================================================

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

from config import (
    SEQUENCE_LENGTH,
    FEATURE_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_LAYERS,
    DROPOUT_RATE,
    CLASSIFIER_THRESHOLD,
)


class BehavioralLSTMClassifier(nn.Module):
    """
    Modèle LSTM pour classification shoplifting/normal.
    Optimisé pour GTX 1650 (4GB VRAM).
    """
    
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_LAYERS,
        dropout: float = DROPOUT_RATE,
    ):
        super(BehavioralLSTMClassifier, self).__init__()
        
        # Encodeur de features (optionnel, pour réduire dimension)
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # LSTM bidirectionnel
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention simple (pas MultiheadAttention pour économiser VRAM)
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        
        # Classificateur final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, sequence_length, input_dim]
            
        Returns:
            output: Tensor [batch_size, 1] - probabilité shoplifting
            attention_weights: Tensor [batch_size, sequence_length, 1]
        """
        # Encoder features
        batch_size, seq_len, _ = x.shape
        x = self.feature_encoder(x)  # [batch, seq, hidden//2]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden*2]
        
        # Attention mechanism simple
        attn_scores = self.attention_weights(lstm_out)  # [batch, seq, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Pooling pondéré par attention
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden*2]
        
        # Classification
        output = self.classifier(context)  # [batch, 1]
        
        return output, attn_weights


class BehaviorSequenceBuffer:
    """
    Buffer pour accumuler les features comportementales et former des séquences.
    """
    
    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        # Buffer par track_id
        self.buffers = {}
    
    def add_features(self, track_id: int, features: np.ndarray):
        """
        Ajoute un vecteur de features pour un track.
        
        Args:
            track_id: ID unique du track
            features: Array numpy [feature_dim]
        """
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.sequence_length)
        
        self.buffers[track_id].append(features)
    
    def get_sequence(self, track_id: int) -> Tuple[bool, np.ndarray]:
        """
        Récupère une séquence complète si disponible.
        
        Returns:
            (is_ready, sequence)
            - is_ready: True si séquence complète
            - sequence: Array numpy [sequence_length, feature_dim] ou None
        """
        if track_id not in self.buffers:
            return False, None
        
        buffer = self.buffers[track_id]
        
        if len(buffer) < self.sequence_length:
            return False, None
        
        # Convertir deque en numpy array
        sequence = np.array(list(buffer))
        
        return True, sequence
    
    def cleanup(self, active_ids: set):
        """
        Nettoie les buffers des tracks inactifs.
        """
        dead_ids = set(self.buffers.keys()) - active_ids
        for track_id in dead_ids:
            del self.buffers[track_id]


class ShopliftingDetector:
    """
    Orchestrateur de la détection de shoplifting en temps réel.
    """
    
    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Modèle
        self.model = BehavioralLSTMClassifier().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
        
        # Buffer de séquences
        self.sequence_buffer = BehaviorSequenceBuffer()
        
        # Résultats de prédiction par track
        self.predictions = {}
    
    def load_model(self, model_path: str):
        """
        Charge les poids du modèle entraîné.
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modèle chargé depuis {model_path}")
    
    def behavior_to_features(self, behavior_data: Dict) -> np.ndarray:
        """
        Convertit les données comportementales en vecteur de features.
        
        Returns:
            Array numpy [feature_dim]
        """
        features = [
            # Features de mouvement (5 dims)
            behavior_data.get('avg_velocity', 0.0),
            behavior_data.get('velocity_std', 0.0),
            behavior_data.get('max_velocity', 0.0),
            behavior_data.get('time_stationary', 0.0),
            behavior_data.get('stationary_ratio', 0.0),
            
            # Gestes suspects (4 dims)
            behavior_data.get('gesture_counts', {}).get('hand_to_pocket', 0),
            behavior_data.get('gesture_counts', {}).get('hand_to_bag', 0),
            behavior_data.get('gesture_counts', {}).get('body_bent', 0),
            behavior_data.get('gesture_counts', {}).get('looking_around', 0),
            
            # Total gestes (1 dim)
            behavior_data.get('total_gestures', 0),
        ]
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, track_id: int, behavior_data: Dict) -> Tuple[bool, float]:
        """
        Fait une prédiction pour un track donné.
        
        Args:
            track_id: ID du track
            behavior_data: Dict avec les features comportementales
            
        Returns:
            (has_prediction, probability)
        """
        # Convertir behavior_data en features
        features = self.behavior_to_features(behavior_data)
        
        # Ajouter au buffer
        self.sequence_buffer.add_features(track_id, features)
        
        # Vérifier si séquence complète
        is_ready, sequence = self.sequence_buffer.get_sequence(track_id)
        
        if not is_ready:
            return False, 0.0
        
        # Prédiction
        with torch.no_grad():
            # Préparer input [1, seq_len, feature_dim]
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Forward
            output, attn_weights = self.model(x)
            
            probability = output.item()
        
        # Stocker résultat
        self.predictions[track_id] = {
            'probability': probability,
            'is_suspicious': probability > CLASSIFIER_THRESHOLD,
            'attention_weights': attn_weights.cpu().numpy()
        }
        
        return True, probability
    
    def get_prediction(self, track_id: int) -> Dict:
        """
        Récupère la dernière prédiction pour un track.
        """
        return self.predictions.get(track_id, {
            'probability': 0.0,
            'is_suspicious': False,
            'attention_weights': None
        })
    
    def cleanup(self, active_ids: set):
        """
        Nettoie les buffers et prédictions des tracks inactifs.
        """
        self.sequence_buffer.cleanup(active_ids)
        
        dead_ids = set(self.predictions.keys()) - active_ids
        for track_id in dead_ids:
            del self.predictions[track_id]


class HeuristicScorer:
    """
    Scoring heuristique simple (baseline sans ML).
    Utile pour tester le pipeline avant l'entraînement du LSTM.
    """
    
    def __init__(self):
        self.scores = {}
    
    def compute_score(self, track_id: int, behavior_data: Dict) -> Tuple[float, str]:
        """
        Calcule un score heuristique basé sur les règles.
        
        Returns:
            (score, risk_level)
        """
        score = 0.0
        
        # Règle 1 : Gestes suspects
        gesture_counts = behavior_data.get('gesture_counts', {})
        score += gesture_counts.get('hand_to_pocket', 0) * 3.0
        score += gesture_counts.get('hand_to_bag', 0) * 4.0
        score += gesture_counts.get('body_bent', 0) * 2.0
        score += gesture_counts.get('looking_around', 0) * 1.5
        
        # Règle 2 : Temps stationnaire anormal (>2 min)
        frames_count = behavior_data.get('frames_count', 0)
        time_stationary = behavior_data.get('time_stationary', 0)
        
        if frames_count > 60 and time_stationary > 60:  # >2 sec stationnaire sur 2 sec
            score += 3.0
        
        # Règle 3 : Vitesse anormale (très lent ou fuite rapide)
        avg_velocity = behavior_data.get('avg_velocity', 0.0)
        max_velocity = behavior_data.get('max_velocity', 0.0)
        
        if avg_velocity < 2.0:  # Trop lent
            score += 2.0
        
        if max_velocity > 15.0:  # Fuite rapide
            score += 5.0
        
        # Règle 4 : Nervosité (forte variation de vitesse)
        velocity_std = behavior_data.get('velocity_std', 0.0)
        if velocity_std > 5.0:
            score += 2.0
        
        # Normaliser score 0-100
        score = min(score, 100.0)
        
        # Risk level
        if score >= 70:
            risk_level = "HIGH_RISK"
        elif score >= 40:
            risk_level = "MEDIUM_RISK"
        else:
            risk_level = "NORMAL"
        
        self.scores[track_id] = {
            'score': score,
            'risk_level': risk_level,
            'probability': score / 100.0
        }
        
        return score, risk_level
    
    def get_score(self, track_id: int) -> Dict:
        """
        Récupère le score d'un track.
        """
        return self.scores.get(track_id, {
            'score': 0.0,
            'risk_level': 'NORMAL',
            'probability': 0.0
        })
    
    def cleanup(self, active_ids: set):
        """
        Nettoie les scores des tracks inactifs.
        """
        dead_ids = set(self.scores.keys()) - active_ids
        for track_id in dead_ids:
            del self.scores[track_id]
