# ============================================================
# detector.py — Détection de personnes avec YOLO26
# ============================================================
# Responsabilités :
#   1. Charger YOLO26n sur le GPU
#   2. Faire l'inférence frame par frame
#   3. Filtrer UNIQUEMENT les personnes (class 0)
#   4. Retourner les détections dans le format DeepSORT
# ============================================================

import numpy as np
from ultralytics import YOLO
from colorama import Fore, Style

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    INFERENCE_SIZE,
    DETECT_CLASS_PERSON,
    NMS_IOU,
)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def nms_xyxy(boxes, scores, iou_thr):
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou_xyxy(boxes[i], boxes[j]) < iou_thr]
    return keep


class YOLODetector:
    """
    Détecte uniquement les personnes dans une frame.
    
    Retourne les détections sous forme de liste pour DeepSORT :
        [ ([left, top, width, height], confidence, "person"), ... ]
    """

    def __init__(self):
        print(f"{Fore.CYAN}[YOLO] Chargement du modèle {MODEL_PATH}...{Style.RESET_ALL}")
        self.model = YOLO(MODEL_PATH)
        self.device = DEVICE
        print(f"{Fore.GREEN}[YOLO] ✅ Modèle chargé sur {self.device.upper()}{Style.RESET_ALL}")

    def detect(self, frame: np.ndarray) -> list:
        """
        Fait l'inférence et retourne les personnes détectées.
        
        Args:
            frame : Image BGR (numpy array)
            
        Returns:
            Liste de tuples dans le format DeepSORT :
                ( [left, top, width, height], confidence, "person" )
            
            ⚠️  DeepSORT attend [left, top, w, h] PAS [x1,y1,x2,y2]
        """
        results = self.model.predict(
            source=frame,
            imgsz=INFERENCE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=self.device,
            classes=[DETECT_CLASS_PERSON],
            stream=True,
            verbose=False,
        )

        detections_for_deepsort = []
        person_boxes = []
        person_scores = []

        for result in results:
            boxes = result.boxes

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                # --- On ne garde QUE les personnes ---
                if class_id != DETECT_CLASS_PERSON:
                    continue

                confidence = float(boxes.conf[i].item())

                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                person_boxes.append([x1, y1, x2, y2])
                person_scores.append(confidence)

        keep = nms_xyxy(person_boxes, person_scores, NMS_IOU)
        for i in keep:
            x1, y1, x2, y2 = person_boxes[i]
            confidence = person_scores[i]
            left = int(x1)
            top = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)
            detections_for_deepsort.append(
                ([left, top, width, height], confidence, "person")
            )

        return detections_for_deepsort
