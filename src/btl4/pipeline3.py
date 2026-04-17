import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Vui lòng cài đặt ultralytics: pip install ultralytics")

try:
    from .base_pipeline import BasePipeline, valid_input
except ImportError:
    from base_pipeline import BasePipeline, valid_input


class ObjectDetectionPipeline(BasePipeline):
    """
    Pipeline so sánh nhận diện người đi bộ: HOG+SVM vs YOLOv8.
    Có hỗ trợ:
    - Chạy detection cho đầu vào ảnh.
    - Đánh giá định lượng trên tập Detection/Test (VOC XML).
    - Trực quan hóa GT, HOG+SVM và YOLOv8 để so sánh.
    """

    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        yolo_conf_threshold: float = 0.25,
        hog_score_threshold: float = 0.0,
        hog_nms_iou_threshold: float = 0.4,
    ):
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.yolo_conf_threshold = float(max(0.0, yolo_conf_threshold))
        self.hog_score_threshold = float(hog_score_threshold)
        self.hog_nms_iou_threshold = float(max(0.0, hog_nms_iou_threshold))

        self.yolo_model_name = yolo_model_name
        self.yolo_model = None
        if YOLO is None:
            print("Không thể tải YOLO vì thiếu thư viện ultralytics.")
        else:
            try:
                self.yolo_model = YOLO(yolo_model_name)
            except Exception as e:
                print(f"Không thể tải mô hình YOLO: {e}")
                self.yolo_model = None

    @staticmethod
    def _to_box_array(boxes: Union[np.ndarray, List[List[float]], List[Tuple[float, float, float, float]], None]) -> np.ndarray:
        if boxes is None:
            return np.zeros((0, 4), dtype=np.float32)
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return arr.reshape(-1, 4)

    @staticmethod
    def _normalize_scores(
        scores: Union[np.ndarray, List[float], Tuple[float, ...], None],
        expected_len: int,
        default_score: float = 1.0,
    ) -> np.ndarray:
        if expected_len <= 0:
            return np.zeros((0,), dtype=np.float32)

        if scores is None:
            return np.full((expected_len,), float(default_score), dtype=np.float32)

        arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        if arr.size == expected_len:
            return arr
        if arr.size < expected_len:
            padding = np.full((expected_len - arr.size,), float(default_score), dtype=np.float32)
            return np.concatenate([arr, padding], axis=0)
        return arr[:expected_len]

    @staticmethod
    def _clip_boxes_xywh(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        arr = ObjectDetectionPipeline._to_box_array(boxes)
        if arr.shape[0] == 0:
            return arr

        clipped = arr.copy()
        x1 = np.clip(clipped[:, 0], 0, max(0, width - 1))
        y1 = np.clip(clipped[:, 1], 0, max(0, height - 1))
        x2 = np.clip(clipped[:, 0] + np.maximum(0.0, clipped[:, 2]), 0, width)
        y2 = np.clip(clipped[:, 1] + np.maximum(0.0, clipped[:, 3]), 0, height)

        clipped[:, 0] = x1
        clipped[:, 1] = y1
        clipped[:, 2] = np.maximum(0.0, x2 - x1)
        clipped[:, 3] = np.maximum(0.0, y2 - y1)

        keep = (clipped[:, 2] > 0.0) & (clipped[:, 3] > 0.0)
        return clipped[keep]

    @staticmethod
    def _bbox_iou_xywh(box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax, ay, aw, ah = [float(v) for v in box_a]
        bx, by, bw, bh = [float(v) for v in box_b]
        if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
            return 0.0

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
        inter_h = max(0.0, min(ay2, by2) - max(ay, by))
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return float(inter_area / union)

    def _detect_hog(self, image: np.ndarray) -> Dict[str, Any]:
        """Thực hiện nhận diện bằng HOG+SVM."""
        boxes, weights = self.hog.detectMultiScale(
            image,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
        )

        boxes = self._to_box_array(boxes)
        scores = self._normalize_scores(weights, len(boxes), default_score=1.0)
        if len(boxes) == 0:
            return {"boxes": boxes, "scores": scores}

        keep = scores >= self.hog_score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        if len(boxes) == 0:
            return {"boxes": boxes, "scores": scores}

        if self.hog_nms_iou_threshold > 0.0:
            nms_indices = cv2.dnn.NMSBoxes(
                bboxes=boxes.tolist(),
                scores=scores.tolist(),
                score_threshold=float(self.hog_score_threshold),
                nms_threshold=float(self.hog_nms_iou_threshold),
            )
            if len(nms_indices) == 0:
                return {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros((0,), dtype=np.float32),
                }
            keep_idx = np.asarray(nms_indices, dtype=np.int32).reshape(-1)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]

        return {"boxes": boxes.astype(np.float32), "scores": scores.astype(np.float32)}

    def _detect_yolo(self, image: np.ndarray) -> Dict[str, Any]:
        """Thực hiện nhận diện bằng YOLO, chỉ giữ class person (class id=0)."""
        if self.yolo_model is None:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
            }

        results = self.yolo_model.predict(
            source=image,
            conf=self.yolo_conf_threshold,
            verbose=False,
        )[0]
        if results.boxes is None or len(results.boxes) == 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
            }

        cls_ids = results.boxes.cls.detach().cpu().numpy().astype(np.int32)
        person_mask = cls_ids == 0
        if not np.any(person_mask):
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
            }

        boxes_xywh = results.boxes.xywh.detach().cpu().numpy()[person_mask]
        scores = results.boxes.conf.detach().cpu().numpy()[person_mask].astype(np.float32)

        cv2_boxes = []
        for x_center, y_center, width, height in boxes_xywh:
            cv2_boxes.append(
                [
                    float(x_center - width / 2.0),
                    float(y_center - height / 2.0),
                    float(width),
                    float(height),
                ]
            )

        boxes = self._to_box_array(cv2_boxes)
        return {"boxes": boxes.astype(np.float32), "scores": scores}

    def _draw_detections(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        color: Tuple[int, int, int],
        label: str,
        scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vẽ khung bao lên ảnh."""
        img_copy = image.copy()
        boxes_arr = self._to_box_array(boxes)
        score_arr = self._normalize_scores(scores, len(boxes_arr), default_score=1.0) if scores is not None else None

        for i, box in enumerate(boxes_arr):
            x, y, w, h = [int(round(v)) for v in box]
            x2 = x + max(0, w)
            y2 = y + max(0, h)
            cv2.rectangle(img_copy, (x, y), (x2, y2), color, 2)
            text = label
            if score_arr is not None and i < len(score_arr):
                text = f"{label}: {float(score_arr[i]):.2f}"
            cv2.putText(
                img_copy,
                text,
                (x, max(14, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
            )
        return img_copy

    @valid_input
    def run(self, input: Union[str, List[str], List[np.ndarray], np.ndarray], visualize: bool = False ,hog=True) -> Dict[str, Any]:
        rgb_images = self._read_input(input)

        hog_results = []
        yolo_results = []
        comparison_images = []

        for img in rgb_images:
            hog_out = self._detect_hog(img)
            yolo_out = self._detect_yolo(img)

            if hog:
                img_with_hog = self._draw_detections(img, hog_out["boxes"], (0, 255, 0), "HOG+SVM", hog_out["scores"])
                img_with_both = self._draw_detections(
                    img_with_hog,
                    yolo_out["boxes"],
                    (255, 0, 0),
                    "YOLOv8",
                    yolo_out["scores"],
                )
            else:
                img_with_both = self._draw_detections(
                    img,
                    yolo_out["boxes"],
                    (255, 0, 0),
                    "YOLOv8",
                    yolo_out["scores"],
                )

            hog_results.append(hog_out)
            yolo_results.append(yolo_out)
            comparison_images.append(img_with_both)

        result = {
            "hog_detections": hog_results,
            "yolo_detections": yolo_results,
            "visualized_images": comparison_images,
            "num_images": len(rgb_images),
        }

        if visualize:
            self.visualize(result, "visualized_images", "Detection Comparison: HOG (Green) vs YOLOv8 (Red)")

        return result

    def _parse_voc_annotation(self, xml_path: Union[str, Path], target_class: str = "person") -> Dict[str, Any]:
        root = ET.parse(str(xml_path)).getroot()
        image_filename = root.findtext("filename")
        if image_filename is None:
            image_filename = f"{Path(xml_path).stem}.png"

        gt_boxes: List[List[float]] = []
        for obj in root.findall("object"):
            class_name = (obj.findtext("name") or "").strip().lower()
            if class_name != target_class.lower():
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            width = max(0.0, xmax - xmin)
            height = max(0.0, ymax - ymin)
            if width > 0.0 and height > 0.0:
                gt_boxes.append([xmin, ymin, width, height])

        return {"image_filename": image_filename, "boxes": self._to_box_array(gt_boxes)}

    def load_detection_test_set(self, test_root: Union[str, Path], target_class: str = "person") -> List[Dict[str, Any]]:
        test_root = Path(test_root)
        ann_dir = test_root / "Annotations"
        img_dir = test_root / "JPEGImages"
        if not ann_dir.exists() or not img_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy Annotations/JPEGImages trong: {test_root}")

        xml_paths = sorted(ann_dir.glob("*.xml"))
        samples: List[Dict[str, Any]] = []
        for xml_path in xml_paths:
            ann = self._parse_voc_annotation(xml_path, target_class=target_class)
            image_filename = ann["image_filename"]

            image_path = img_dir / image_filename
            if not image_path.exists():
                image_path = img_dir / f"{xml_path.stem}.png"
            if not image_path.exists():
                image_path = img_dir / f"{xml_path.stem}.jpg"
            if not image_path.exists():
                continue

            samples.append(
                {
                    "image_id": xml_path.stem,
                    "image_path": str(image_path),
                    "gt_boxes": ann["boxes"],
                }
            )

        if len(samples) == 0:
            raise ValueError(f"Không tìm thấy sample hợp lệ trong: {test_root}")
        return samples

    @staticmethod
    def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
        if recall.size == 0 or precision.size == 0:
            return 0.0

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        return float(ap)

    def _evaluate_at_iou(
        self,
        gt_boxes_per_image: List[np.ndarray],
        pred_boxes_per_image: List[np.ndarray],
        pred_scores_per_image: List[np.ndarray],
        iou_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        total_gt = int(sum(len(gt_boxes) for gt_boxes in gt_boxes_per_image))

        predictions: List[Tuple[int, float, np.ndarray]] = []
        for image_idx, (boxes, scores) in enumerate(zip(pred_boxes_per_image, pred_scores_per_image)):
            boxes_arr = self._to_box_array(boxes)
            scores_arr = self._normalize_scores(scores, len(boxes_arr), default_score=1.0)
            for box, score in zip(boxes_arr, scores_arr):
                predictions.append((image_idx, float(score), box.astype(np.float32)))

        predictions.sort(key=lambda item: item[1], reverse=True)

        matched_gt_flags = [np.zeros(len(gt_boxes), dtype=bool) for gt_boxes in gt_boxes_per_image]
        true_positive: List[float] = []
        false_positive: List[float] = []
        matched_ious: List[float] = []

        for image_idx, _, pred_box in predictions:
            gt_boxes = gt_boxes_per_image[image_idx]
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if matched_gt_flags[image_idx][gt_idx]:
                    continue
                iou = self._bbox_iou_xywh(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                matched_gt_flags[image_idx][best_gt_idx] = True
                true_positive.append(1.0)
                false_positive.append(0.0)
                matched_ious.append(best_iou)
            else:
                true_positive.append(0.0)
                false_positive.append(1.0)

        if len(true_positive) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "ap": 0.0,
                "mean_iou": 0.0,
                "num_predictions": 0,
                "num_ground_truth": total_gt,
                "tp": 0.0,
                "fp": 0.0,
                "fn": float(total_gt),
                "precision_curve": np.array([], dtype=np.float32),
                "recall_curve": np.array([], dtype=np.float32),
            }

        tp_arr = np.asarray(true_positive, dtype=np.float32)
        fp_arr = np.asarray(false_positive, dtype=np.float32)

        tp_cumsum = np.cumsum(tp_arr)
        fp_cumsum = np.cumsum(fp_arr)
        precision_curve = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
        recall_curve = tp_cumsum / max(float(total_gt), 1e-12)
        ap = self._compute_ap(recall_curve, precision_curve)

        tp_total = float(tp_cumsum[-1])
        fp_total = float(fp_cumsum[-1])
        fn_total = float(max(0.0, total_gt - tp_total))
        precision = tp_total / max(tp_total + fp_total, 1e-12)
        recall = tp_total / max(float(total_gt), 1e-12) if total_gt > 0 else 0.0
        mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "ap": float(ap),
            "mean_iou": float(mean_iou),
            "num_predictions": int(len(predictions)),
            "num_ground_truth": int(total_gt),
            "tp": float(tp_total),
            "fp": float(fp_total),
            "fn": float(fn_total),
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
        }

    def _evaluate_detector_metrics(
        self,
        gt_boxes_per_image: List[np.ndarray],
        pred_boxes_per_image: List[np.ndarray],
        pred_scores_per_image: List[np.ndarray],
        iou_threshold: float = 0.5,
        map_iou_thresholds: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        if map_iou_thresholds is None:
            map_iou_thresholds = [float(np.round(t, 2)) for t in np.arange(0.5, 1.0, 0.05)]

        eval_main = self._evaluate_at_iou(
            gt_boxes_per_image=gt_boxes_per_image,
            pred_boxes_per_image=pred_boxes_per_image,
            pred_scores_per_image=pred_scores_per_image,
            iou_threshold=float(iou_threshold),
        )

        ap_values = []
        for thr in map_iou_thresholds:
            eval_at_thr = self._evaluate_at_iou(
                gt_boxes_per_image=gt_boxes_per_image,
                pred_boxes_per_image=pred_boxes_per_image,
                pred_scores_per_image=pred_scores_per_image,
                iou_threshold=float(thr),
            )
            ap_values.append(float(eval_at_thr["ap"]))

        map_score = float(np.mean(ap_values)) if len(ap_values) > 0 else 0.0
        return {
            "iou": eval_main["mean_iou"],
            "precision": eval_main["precision"],
            "recall": eval_main["recall"],
            "ap": eval_main["ap"],
            "mAP": map_score,
            "iou_eval_threshold": float(iou_threshold),
            "map_iou_thresholds": map_iou_thresholds,
            "num_predictions": eval_main["num_predictions"],
            "num_ground_truth": eval_main["num_ground_truth"],
            "tp": eval_main["tp"],
            "fp": eval_main["fp"],
            "fn": eval_main["fn"],
        }

    @staticmethod
    def _format_metrics_table(hog_metrics: Dict[str, Any], yolo_metrics: Dict[str, Any]) -> str:
        headers = ["Thuật toán", "IoU", "Precision", "Recall", "mAP"]
        rows = [
            [
                "HOG + SVM",
                f"{hog_metrics['iou']:.4f}",
                f"{hog_metrics['precision']:.4f}",
                f"{hog_metrics['recall']:.4f}",
                f"{hog_metrics['mAP']:.4f}",
            ],
            [
                "YOLOv8",
                f"{yolo_metrics['iou']:.4f}",
                f"{yolo_metrics['precision']:.4f}",
                f"{yolo_metrics['recall']:.4f}",
                f"{yolo_metrics['mAP']:.4f}",
            ],
        ]

        col_widths = []
        for i, head in enumerate(headers):
            width = max(len(head), max(len(row[i]) for row in rows))
            col_widths.append(width)

        def fmt_row(row_vals: List[str]) -> str:
            cells = [f"{v:<{col_widths[i]}}" for i, v in enumerate(row_vals)]
            return "| " + " | ".join(cells) + " |"

        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        lines = [sep, fmt_row(headers), sep]
        for row in rows:
            lines.append(fmt_row(row))
        lines.append(sep)
        return "\n".join(lines)

    def visualize_test_predictions(
        self,
        records: List[Dict[str, Any]],
        sample_count: int = 6,
        show: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if len(records) == 0:
            return {"selected_indices": [], "selected_image_ids": [], "save_path": None}

        n_samples = int(max(1, min(sample_count, len(records))))
        selected_indices = np.linspace(0, len(records) - 1, num=n_samples, dtype=int).tolist()

        fig, axes = plt.subplots(n_samples, 4, figsize=(22, 5 * n_samples))
        if n_samples == 1:
            axes = np.expand_dims(axes, axis=0)

        titles = ["Input", "Ground Truth", "HOG + SVM", "YOLOv8"]
        for col_idx, title in enumerate(titles):
            axes[0, col_idx].set_title(title, fontsize=12)

        selected_ids = []
        for row_idx, sample_idx in enumerate(selected_indices):
            rec = records[sample_idx]
            selected_ids.append(rec["image_id"])
            image_rgb = rec["image_rgb"]

            gt_img = self._draw_detections(image_rgb, rec["gt_boxes"], (255, 255, 0), "GT")
            hog_img = self._draw_detections(
                image_rgb,
                rec["hog_boxes"],
                (0, 255, 0),
                "HOG+SVM",
                rec["hog_scores"],
            )
            yolo_img = self._draw_detections(
                image_rgb,
                rec["yolo_boxes"],
                (255, 0, 0),
                "YOLOv8",
                rec["yolo_scores"],
            )

            row_images = [image_rgb, gt_img, hog_img, yolo_img]
            for col_idx, vis_img in enumerate(row_images):
                axes[row_idx, col_idx].imshow(vis_img)
                axes[row_idx, col_idx].axis("off")
                if col_idx == 0:
                    axes[row_idx, col_idx].set_ylabel(rec["image_id"], fontsize=10)

        plt.tight_layout()

        saved_path = None
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
            saved_path = str(save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return {
            "selected_indices": selected_indices,
            "selected_image_ids": selected_ids,
            "save_path": saved_path,
        }

    def evaluate_on_detection_test(
        self,
        test_root: Optional[Union[str, Path]] = None,
        iou_threshold: float = 0.5,
        map_iou_thresholds: Optional[List[float]] = None,
        visualize: bool = True,
        sample_count: int = 6,
        save_visual_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if test_root is None:
            test_root = _default_detection_test_dir()

        samples = self.load_detection_test_set(test_root=test_root, target_class="person")

        gt_boxes_per_image: List[np.ndarray] = []
        hog_boxes_per_image: List[np.ndarray] = []
        hog_scores_per_image: List[np.ndarray] = []
        yolo_boxes_per_image: List[np.ndarray] = []
        yolo_scores_per_image: List[np.ndarray] = []

        vis_records: List[Dict[str, Any]] = []
        total_samples = len(samples)

        for idx, sample in enumerate(samples, start=1):
            image_bgr = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Không đọc được ảnh: {sample['image_path']}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]

            gt_boxes = self._clip_boxes_xywh(sample["gt_boxes"], width=w, height=h)

            hog_out = self._detect_hog(image_rgb)
            yolo_out = self._detect_yolo(image_rgb)

            hog_boxes = self._clip_boxes_xywh(hog_out["boxes"], width=w, height=h)
            yolo_boxes = self._clip_boxes_xywh(yolo_out["boxes"], width=w, height=h)
            hog_scores = self._normalize_scores(hog_out.get("scores"), len(hog_boxes), default_score=1.0)
            yolo_scores = self._normalize_scores(yolo_out.get("scores"), len(yolo_boxes), default_score=1.0)

            gt_boxes_per_image.append(gt_boxes)
            hog_boxes_per_image.append(hog_boxes)
            hog_scores_per_image.append(hog_scores)
            yolo_boxes_per_image.append(yolo_boxes)
            yolo_scores_per_image.append(yolo_scores)

            vis_records.append(
                {
                    "image_id": sample["image_id"],
                    "image_rgb": image_rgb,
                    "gt_boxes": gt_boxes,
                    "hog_boxes": hog_boxes,
                    "hog_scores": hog_scores,
                    "yolo_boxes": yolo_boxes,
                    "yolo_scores": yolo_scores,
                }
            )

            if idx % 50 == 0 or idx == total_samples:
                print(f"Đã xử lý {idx}/{total_samples} ảnh...")

        hog_metrics = self._evaluate_detector_metrics(
            gt_boxes_per_image=gt_boxes_per_image,
            pred_boxes_per_image=hog_boxes_per_image,
            pred_scores_per_image=hog_scores_per_image,
            iou_threshold=iou_threshold,
            map_iou_thresholds=map_iou_thresholds,
        )
        yolo_metrics = self._evaluate_detector_metrics(
            gt_boxes_per_image=gt_boxes_per_image,
            pred_boxes_per_image=yolo_boxes_per_image,
            pred_scores_per_image=yolo_scores_per_image,
            iou_threshold=iou_threshold,
            map_iou_thresholds=map_iou_thresholds,
        )

        metrics_table = self._format_metrics_table(hog_metrics, yolo_metrics)
        map_thr = yolo_metrics["map_iou_thresholds"]
        map_thr_str = ", ".join(f"{float(t):.2f}" for t in map_thr)

        print("\nKết quả đánh giá Object Detection trên Detection/Test")
        print(f"IoU threshold cho Precision/Recall: {float(iou_threshold):.2f}")
        print(f"mAP tính trung bình AP trên các ngưỡng IoU: [{map_thr_str}]")
        print(metrics_table)

        vis_meta = None
        if visualize:
            if save_visual_path is None:
                save_visual_path = Path(test_root) / "comparison_hog_yolov8_groundtruth.png"
            vis_meta = self.visualize_test_predictions(
                records=vis_records,
                sample_count=sample_count,
                show=True,
                save_path=save_visual_path,
            )
            if vis_meta["save_path"] is not None:
                print(f"Đã lưu ảnh trực quan tại: {vis_meta['save_path']}")

        return {
            "dataset_root": str(test_root),
            "num_images": len(samples),
            "hog_metrics": hog_metrics,
            "yolo_metrics": yolo_metrics,
            "metrics_table": metrics_table,
            "visualization": vis_meta,
        }


def _default_detection_test_dir() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "img" / "btl4" / "Detection" / "Test")


if __name__ == "__main__":
    detection_test_dir = _default_detection_test_dir()
    if not os.path.isdir(detection_test_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {detection_test_dir}")

    pipeline = ObjectDetectionPipeline(
        yolo_model_name="yolov8n.pt",
        yolo_conf_threshold=0.25,
        hog_score_threshold=0.0,
        hog_nms_iou_threshold=0.4,
    )

    pipeline.evaluate_on_detection_test(
        test_root=detection_test_dir,
        iou_threshold=0.5,
        map_iou_thresholds=[float(np.round(t, 2)) for t in np.arange(0.5, 1.0, 0.05)],
        visualize=True,
        sample_count=6,
        save_visual_path=Path(detection_test_dir) / "comparison_hog_yolov8_groundtruth.png",
    )
