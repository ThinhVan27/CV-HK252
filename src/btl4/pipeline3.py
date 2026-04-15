import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Union, Optional
try:
    from ultralytics import YOLO
except ImportError:
    print("Vui lòng cài đặt ultralytics: pip install ultralytics")

try:
    from .base_pipeline import BasePipeline, valid_input
except ImportError:
    from base_pipeline import BasePipeline, valid_input

class ObjectDetectionPipeline(BasePipeline):
    """
    Pipeline so sánh nhận diện người đi bộ: HOG+SVM vs YOLO.
    """
    def __init__(self, yolo_model_name: str = "yolov8n.pt"):
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 128),   
            _blockSize=(16, 16),    
            _blockStride=(8, 8),     
            _cellSize=(8, 8),        
            _nbins=9                 
        )
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        try:
            self.yolo_model = YOLO(yolo_model_name)
        except Exception as e:
            print(f"Không thể tải mô hình YOLO: {e}")
            self.yolo_model = None

    def _detect_hog(self, image: np.ndarray) -> Dict[str, Any]:
        """Thực hiện nhận diện bằng HOG+SVM."""
        boxes, weights = self.hog.detectMultiScale(
            image, 
            winStride=(8, 8), 
            padding=(16, 16),
            scale=1.05
        )
        return {"boxes": boxes, "scores": weights}

    def _detect_yolo(self, image: np.ndarray) -> Dict[str, Any]:
        """Thực hiện nhận diện bằng YOLO."""
        if self.yolo_model is None:
            return {"boxes": [], "scores": []}
        
        results = self.yolo_model(image, verbose=False)[0]
        person_indices = results.boxes.cls == 0
        boxes = results.boxes.xywh[person_indices].cpu().numpy()
        scores = results.boxes.conf[person_indices].cpu().numpy()
        
        cv2_boxes = []
        for b in boxes:
            x_center, y_center, w, h = b
            cv2_boxes.append([int(x_center - w/2), int(y_center - h/2), int(w), int(h)])
            
        return {"boxes": np.array(cv2_boxes), "scores": scores}

    def _draw_detections(self, image: np.ndarray, boxes: np.ndarray, color: tuple, label: str):
        """Vẽ khung bao lên ảnh."""
        img_copy = image.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_copy, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img_copy

    @valid_input
    def run(self, input: Union[str, List[str], List[np.ndarray], np.ndarray], visualize: bool = False) -> Dict[str, Any]:
        rgb_images = self._read_input(input)
        
        hog_results = []
        yolo_results = []
        comparison_images = []

        for img in rgb_images:
            # Chạy cả hai thuật toán
            hog_out = self._detect_hog(img)
            yolo_out = self._detect_yolo(img)

            img_with_hog = self._draw_detections(img, hog_out["boxes"], (0, 255, 0), "HOG (2005)")
            img_with_both = self._draw_detections(img_with_hog, yolo_out["boxes"], (255, 0, 0), "YOLO")

            hog_results.append(hog_out)
            yolo_results.append(yolo_out)
            comparison_images.append(img_with_both)

        result = {
            "hog_detections": hog_results,
            "yolo_detections": yolo_results,
            "visualized_images": comparison_images,
            "num_images": len(rgb_images)
        }

        if visualize:
            self.visualize(result, "visualized_images", "Detection Comparison: HOG (Green) vs YOLO (Red)")

        return result