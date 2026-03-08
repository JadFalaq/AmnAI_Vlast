# ============================================================
# train_model.py – Entraînement du classificateur LSTM
# ============================================================
# Usage:
#   python train_model.py --shoplifting_dir videos/shoplifting --normal_dir videos/normal
# ============================================================

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

from detector import YOLODetector
from tracker import PersonTracker
from behavior_analyzer import TrackBehaviorAnalyzer
from behavior_classifier import BehavioralLSTMClassifier
from config import (
    SEQUENCE_LENGTH,
    FEATURE_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_LAYERS,
    DROPOUT_RATE,
)


class ShopliftingDataset(Dataset):
    """
    Dataset pour l'entraînement du modèle.
    Chaque sample = séquence de features d'un track + label
    """
    
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return sequence, label


def extract_features_from_video(video_path, detector, tracker, analyzer, label):
    """
    Extrait les séquences de features comportementales d'une vidéo.
    
    Args:
        video_path: Chemin vers la vidéo
        detector: YOLODetector
        tracker: PersonTracker
        analyzer: TrackBehaviorAnalyzer
        label: 0 (normal) ou 1 (shoplifting)
        
    Returns:
        Liste de (sequence, label)
    """
    print(f"   Traitement : {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"   ❌ Impossible d'ouvrir {video_path}")
        return []
    
    sequences_data = []
    features_buffer = {}  # track_id → liste de features
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Détection YOLO
        detections = detector.detect(frame)
        
        # Tracking DeepSORT
        tracked_persons = tracker.update(detections, frame)
        
        # Analyse comportementale
        behavior_results = analyzer.update(frame, tracked_persons)
        
        # Convertir behavior en features
        for track_id, behavior_data in behavior_results.items():
            features = behavior_to_features(behavior_data)
            
            if track_id not in features_buffer:
                features_buffer[track_id] = []
            
            features_buffer[track_id].append(features)
            
            # Si séquence complète → sauvegarder
            if len(features_buffer[track_id]) >= SEQUENCE_LENGTH:
                sequence = np.array(features_buffer[track_id][-SEQUENCE_LENGTH:])
                sequences_data.append((sequence, label))
                
                # Sliding window : supprimer la moitié pour créer overlap
                features_buffer[track_id] = features_buffer[track_id][SEQUENCE_LENGTH // 2:]
    
    cap.release()
    
    print(f"   ✅ {len(sequences_data)} séquences extraites ({frame_count} frames)")
    
    return sequences_data


def behavior_to_features(behavior_data):
    """
    Convertit behavior_data en vecteur de features.
    """
    features = [
        behavior_data.get('avg_velocity', 0.0),
        behavior_data.get('velocity_std', 0.0),
        behavior_data.get('max_velocity', 0.0),
        behavior_data.get('time_stationary', 0.0),
        behavior_data.get('stationary_ratio', 0.0),
        behavior_data.get('gesture_counts', {}).get('hand_to_pocket', 0),
        behavior_data.get('gesture_counts', {}).get('hand_to_bag', 0),
        behavior_data.get('gesture_counts', {}).get('body_bent', 0),
        behavior_data.get('gesture_counts', {}).get('looking_around', 0),
        behavior_data.get('total_gestures', 0),
    ]
    
    return np.array(features, dtype=np.float32)


def train_model(
    shoplifting_dir,
    normal_dir,
    output_model_path="model_shoplifting.pth",
    epochs=50,
    batch_size=16,
    learning_rate=0.001,
):
    """
    Entraîne le modèle LSTM sur les vidéos shoplifting/normal.
    """
    print("\n" + "="*60)
    print("🎓 ENTRAÎNEMENT DU MODÈLE SHOPLIFTING")
    print("="*60 + "\n")
    
    # Initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device : {device}")
    
    detector = YOLODetector()
    
    # ========== EXTRACTION DES FEATURES ==========
    print("\n📊 Extraction des features des vidéos...")
    
    all_sequences = []
    all_labels = []
    
    # Vidéos shoplifting (label = 1)
    print(f"\n🚨 Vidéos SHOPLIFTING : {shoplifting_dir}")
    shoplifting_videos = [
        os.path.join(shoplifting_dir, f)
        for f in os.listdir(shoplifting_dir)
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]
    
    for video_path in shoplifting_videos:
        tracker = PersonTracker()
        analyzer = TrackBehaviorAnalyzer()
        
        sequences_data = extract_features_from_video(
            video_path, detector, tracker, analyzer, label=1
        )
        
        for seq, lbl in sequences_data:
            all_sequences.append(seq)
            all_labels.append(lbl)
    
    # Vidéos normales (label = 0)
    print(f"\n✅ Vidéos NORMALES : {normal_dir}")
    normal_videos = [
        os.path.join(normal_dir, f)
        for f in os.listdir(normal_dir)
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]
    
    for video_path in normal_videos:
        tracker = PersonTracker()
        analyzer = TrackBehaviorAnalyzer()
        
        sequences_data = extract_features_from_video(
            video_path, detector, tracker, analyzer, label=0
        )
        
        for seq, lbl in sequences_data:
            all_sequences.append(seq)
            all_labels.append(lbl)
    
    print(f"\n📈 Total séquences : {len(all_sequences)}")
    print(f"   - Shoplifting : {sum(all_labels)}")
    print(f"   - Normal : {len(all_labels) - sum(all_labels)}")
    
    if len(all_sequences) == 0:
        print("❌ Aucune séquence extraite ! Vérifiez vos vidéos.")
        return
    
    # ========== SPLIT TRAIN/VAL/TEST ==========
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📦 Dataset split :")
    print(f"   Train : {len(X_train)} séquences")
    print(f"   Val   : {len(X_val)} séquences")
    print(f"   Test  : {len(X_test)} séquences")
    
    # ========== DATALOADERS ==========
    train_dataset = ShopliftingDataset(X_train, y_train)
    val_dataset = ShopliftingDataset(X_val, y_val)
    test_dataset = ShopliftingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ========== MODÈLE ==========
    model = BehavioralLSTMClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_LAYERS,
        dropout=DROPOUT_RATE,
    ).to(device)
    
    print(f"\n🤖 Modèle : {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Loss avec pondération (shoplifting rare)
    num_normal = len(all_labels) - sum(all_labels)
    num_shoplifting = sum(all_labels)
    weight = torch.tensor([num_normal / num_shoplifting]).to(device)
    
    criterion = nn.BCELoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # ========== ENTRAÎNEMENT ==========
    print(f"\n🏋️ Entraînement ({epochs} epochs)...\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs, _ = model(sequences)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs, _ = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total
        
        # Scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_accuracy:.2f}%")
        
        # Sauvegarder meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
            }, output_model_path)
            print(f"   ✅ Modèle sauvegardé (val_loss={avg_val_loss:.4f})")
    
    # ========== TEST ==========
    print(f"\n🧪 Évaluation sur test set...")
    
    model.load_state_dict(torch.load(output_model_path)['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(sequences)
            predictions = (outputs > 0.5).float()
            
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
    
    test_accuracy = 100.0 * test_correct / test_total
    
    print(f"\n✅ Test Accuracy : {test_accuracy:.2f}%")
    print(f"\n💾 Modèle final sauvegardé : {output_model_path}")
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Entraîner le modèle shoplifting")
    parser.add_argument(
        "--shoplifting_dir",
        type=str,
        required=True,
        help="Dossier contenant les vidéos de shoplifting"
    )
    parser.add_argument(
        "--normal_dir",
        type=str,
        required=True,
        help="Dossier contenant les vidéos normales"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_shoplifting.pth",
        help="Chemin de sauvegarde du modèle"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    train_model(
        shoplifting_dir=args.shoplifting_dir,
        normal_dir=args.normal_dir,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
