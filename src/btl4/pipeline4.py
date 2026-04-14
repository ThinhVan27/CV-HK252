import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Optional, Tuple
from abc import ABC, abstractmethod
from collections import deque

from sklearn.cluster import KMeans

try:
    from .base_pipeline import *
except ImportError:
    from base_pipeline import *


class Segmenter(ABC):
    """Base interface for segmentation algorithms."""

    @abstractmethod
    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError


class RegionGrowingSegmenter(Segmenter):
    """Region growing on grayscale image from one or multiple seeds."""

    def __init__(
        self,
        threshold: int = 5,
        connectivity: int = 4,
        seeds: Optional[List[Tuple[int, int]]] = None,
    ):
        self.threshold = int(max(0, threshold))
        self.connectivity = 8 if connectivity == 8 else 4
        self.seeds = seeds

    def _neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        if self.connectivity == 8:
            return [
                (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                (x - 1, y),                 (x + 1, y),
                (x - 1, y + 1), (x, y + 1), (x + 1, y + 1),
            ]
        return [(x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)]

    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        if input.ndim == 3:
            gray = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
            rgb = input
        else:
            gray = input
            rgb = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)

        h, w = gray.shape
        seeds = self.seeds if self.seeds else [(w // 2, h // 2)]
        mask = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)

        for sx, sy in seeds:
            if not (0 <= sx < w and 0 <= sy < h):
                continue
            seed_val = int(gray[sy, sx])
            q: deque = deque([(sx, sy)])

            while q:
                x, y = q.popleft()
                if visited[y, x]:
                    continue
                visited[y, x] = True

                if abs(int(gray[y, x]) - seed_val) <= self.threshold:
                    mask[y, x] = 255
                    for nx, ny in self._neighbors(x, y):
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                            q.append((nx, ny))

        overlay = rgb.copy()
        overlay[mask > 0] = np.array([255, 0, 0], dtype=np.uint8)
        mixed = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)

        return {
            "mask": mask,
            "labels": (mask > 0).astype(np.int32),
            "segments_image": mixed,
            "num_segments": int((mask > 0).any()),
        }


class SplitMergeSegmenter(Segmenter):
    """Quadtree-like split and region-adjacency merge on grayscale image."""

    def __init__(
        self,
        split_std_threshold: float = 8.0,
        merge_mean_threshold: float = 6.0,
        min_region_size: int = 8,
    ):
        self.split_std_threshold = float(max(0.0, split_std_threshold))
        self.merge_mean_threshold = float(max(0.0, merge_mean_threshold))
        self.min_region_size = int(max(2, min_region_size))

    def _split(self, gray: np.ndarray, x0: int, y0: int, w: int, h: int, regions: List[Dict[str, Any]]) -> None:
        patch = gray[y0:y0 + h, x0:x0 + w]
        if patch.size == 0:
            return

        std = float(np.std(patch))
        if std <= self.split_std_threshold or w <= self.min_region_size or h <= self.min_region_size:
            regions.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "w": w,
                    "h": h,
                    "mean": float(np.mean(patch)),
                }
            )
            return

        w2 = w // 2
        h2 = h // 2
        if w2 <= 0 or h2 <= 0:
            regions.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "w": w,
                    "h": h,
                    "mean": float(np.mean(patch)),
                }
            )
            return

        self._split(gray, x0, y0, w2, h2, regions)
        self._split(gray, x0 + w2, y0, w - w2, h2, regions)
        self._split(gray, x0, y0 + h2, w2, h - h2, regions)
        self._split(gray, x0 + w2, y0 + h2, w - w2, h - h2, regions)

    @staticmethod
    def _touch(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        ax1, ay1, ax2, ay2 = a["x0"], a["y0"], a["x0"] + a["w"], a["y0"] + a["h"]
        bx1, by1, bx2, by2 = b["x0"], b["y0"], b["x0"] + b["w"], b["y0"] + b["h"]

        overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
        overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
        vertical_touch = (ax2 == bx1 or bx2 == ax1) and overlap_y > 0
        horizontal_touch = (ay2 == by1 or by2 == ay1) and overlap_x > 0
        return vertical_touch or horizontal_touch

    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        if input.ndim == 3:
            gray = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
            rgb = input
        else:
            gray = input
            rgb = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)

        h, w = gray.shape
        regions: List[Dict[str, Any]] = []
        self._split(gray, 0, 0, w, h, regions)

        n = len(regions)
        parent = np.arange(n, dtype=np.int32)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(n):
            for j in range(i + 1, n):
                if self._touch(regions[i], regions[j]):
                    if abs(regions[i]["mean"] - regions[j]["mean"]) <= self.merge_mean_threshold:
                        union(i, j)

        root_to_label: Dict[int, int] = {}
        labels = np.zeros((h, w), dtype=np.int32)
        next_label = 1

        for idx, r in enumerate(regions):
            root = find(idx)
            if root not in root_to_label:
                root_to_label[root] = next_label
                next_label += 1
            lab = root_to_label[root]
            x0, y0, rw, rh = r["x0"], r["y0"], r["w"], r["h"]
            labels[y0:y0 + rh, x0:x0 + rw] = lab

        num_labels = int(labels.max())
        rng = np.random.default_rng(42)
        palette = np.zeros((max(1, num_labels) + 1, 3), dtype=np.uint8)
        if num_labels > 0:
            palette[1:] = rng.integers(0, 256, size=(num_labels, 3), dtype=np.uint8)
        segments = palette[labels]
        mixed = cv2.addWeighted(rgb, 0.55, segments, 0.45, 0)

        return {
            "mask": labels.astype(np.uint8),
            "labels": labels,
            "segments_image": mixed,
            "num_segments": num_labels,
        }


class KMeansSegmenter(Segmenter):
    """K-means segmentation using color or color+position features."""

    def __init__(
        self,
        k: int = 5,
        use_position: bool = False,
        normalize: bool = True,
        random_state: int = 42,
        max_iter: int = 300,
    ):
        self.k = int(max(2, k))
        self.use_position = bool(use_position)
        self.normalize = bool(normalize)
        self.random_state = int(random_state)
        self.max_iter = int(max(10, max_iter))

    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        if input.ndim != 3:
            rgb = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
        else:
            rgb = input

        h, w = rgb.shape[:2]
        rgb_f = rgb.astype(np.float32) / 255.0
        feat = rgb_f.reshape(-1, 3)

        if self.use_position:
            yy, xx = np.mgrid[0:h, 0:w]
            pos = np.stack([xx / max(1, w - 1), yy / max(1, h - 1)], axis=-1).reshape(-1, 2)
            feat = np.concatenate([feat, pos], axis=1)

        if self.normalize:
            mean = feat.mean(axis=0, keepdims=True)
            std = feat.std(axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            feat = (feat - mean) / std

        km = KMeans(
            n_clusters=self.k,
            n_init=10,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        labels = km.fit_predict(feat).reshape(h, w)

        color_centroids = np.zeros((self.k, 3), dtype=np.uint8)
        rgb_flat = rgb.reshape(-1, 3)
        labels_flat = labels.reshape(-1)
        for c in range(self.k):
            idx = labels_flat == c
            if np.any(idx):
                color_centroids[c] = np.mean(rgb_flat[idx], axis=0).astype(np.uint8)

        segments = color_centroids[labels]

        return {
            "mask": labels.astype(np.uint8),
            "labels": labels.astype(np.int32),
            "segments_image": segments,
            "num_segments": int(np.unique(labels).size),
        }


class SAMSegmenter(Segmenter):
    """Segmentation with Ultralytics SAM."""

    def __init__(self, model_name: str = "sam_b.pt"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import SAM
            except Exception as exc:
                raise ImportError(
                    "Ultralytics SAM is not installed. Install with `pip install ultralytics`."
                ) from exc
            self._model = SAM(self.model_name)

    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        if input.ndim != 3:
            rgb = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
        else:
            rgb = input

        self._load_model()
        results = self._model(rgb)
        if not results or getattr(results[0], "masks", None) is None:
            h, w = rgb.shape[:2]
            empty_labels = np.zeros((h, w), dtype=np.int32)
            return {
                "mask": empty_labels.astype(np.uint8),
                "labels": empty_labels,
                "segments_image": rgb.copy(),
                "num_segments": 0,
            }

        masks = results[0].masks.data.cpu().numpy().astype(bool)
        h, w = rgb.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)

        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        order = np.argsort(-areas)

        for seg_id, idx in enumerate(order, start=1):
            labels[masks[idx]] = seg_id

        n_seg = int(labels.max())
        rng = np.random.default_rng(123)
        palette = np.zeros((max(1, n_seg) + 1, 3), dtype=np.uint8)
        if n_seg > 0:
            palette[1:] = rng.integers(0, 256, size=(n_seg, 3), dtype=np.uint8)
        colorized = palette[labels]
        mixed = cv2.addWeighted(rgb, 0.55, colorized, 0.45, 0)

        return {
            "mask": labels.astype(np.uint8),
            "labels": labels,
            "segments_image": mixed,
            "num_segments": n_seg,
        }

 
class SegmentationPipeline(BasePipeline):
    """
    Segmentation Pipeline
    """

    def __init__(
        self,
        method: str = "kmeans",
        region_growing_threshold: int = 5,
        region_connectivity: int = 4,
        split_std_threshold: float = 8.0,
        merge_mean_threshold: float = 6.0,
        split_min_region_size: int = 8,
        kmeans_k: int = 5,
        kmeans_use_position: bool = False,
        kmeans_normalize: bool = True,
        sam_model_name: str = "sam_b.pt",
        segmenter: Optional[Segmenter] = None,
    ):
        self.method = method.strip().lower()
        self.segmenter = segmenter or self._build_segmenter(
            method=self.method,
            region_growing_threshold=region_growing_threshold,
            region_connectivity=region_connectivity,
            split_std_threshold=split_std_threshold,
            merge_mean_threshold=merge_mean_threshold,
            split_min_region_size=split_min_region_size,
            kmeans_k=kmeans_k,
            kmeans_use_position=kmeans_use_position,
            kmeans_normalize=kmeans_normalize,
            sam_model_name=sam_model_name,
        )

    @staticmethod
    def _build_segmenter(
        method: str,
        region_growing_threshold: int,
        region_connectivity: int,
        split_std_threshold: float,
        merge_mean_threshold: float,
        split_min_region_size: int,
        kmeans_k: int,
        kmeans_use_position: bool,
        kmeans_normalize: bool,
        sam_model_name: str,
    ) -> Segmenter:
        if method == "region_growing":
            return RegionGrowingSegmenter(
                threshold=region_growing_threshold,
                connectivity=region_connectivity,
            )
        if method == "split_merge":
            return SplitMergeSegmenter(
                split_std_threshold=split_std_threshold,
                merge_mean_threshold=merge_mean_threshold,
                min_region_size=split_min_region_size,
            )
        if method == "kmeans":
            return KMeansSegmenter(
                k=kmeans_k,
                use_position=kmeans_use_position,
                normalize=kmeans_normalize,
            )
        if method == "sam":
            return SAMSegmenter(model_name=sam_model_name)
        raise ValueError("Unknown segmentation method. Use one of: region_growing, split_merge, kmeans, sam")

    @valid_input
    def run(self, input: Union[str, List[str], List[np.ndarray], np.ndarray], visualize: bool = False) -> Dict[str, Any]:
        """
        Run `SegmentationPipeline`.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        rgb_images = self._read_input(input)

        masks_images: List[np.ndarray] = []
        labels_images: List[np.ndarray] = []
        segments_images: List[np.ndarray] = []
        metrics: List[Dict[str, Any]] = []

        for rgb in rgb_images:
            out = self.segmenter.extract(rgb)
            mask = out["mask"].astype(np.uint8)
            labels = out["labels"].astype(np.int32)
            seg_img = out["segments_image"].astype(np.uint8)
            num_seg = int(out.get("num_segments", np.unique(labels).size))

            masks_images.append(mask)
            labels_images.append(labels)
            segments_images.append(seg_img)
            metrics.append(
                {
                    "num_segments": num_seg,
                    "foreground_ratio": float(np.count_nonzero(mask)) / float(mask.size),
                }
            )

        result: Dict[str, Any] = {
            "method": self.method,
            "rgb_images": rgb_images,
            "masks_images": masks_images,
            "labels_images": labels_images,
            "segments_images": segments_images,
            "metrics": metrics,
            "num_images": len(rgb_images),
        }

        if visualize:
            self.visualize(result, "masks_images", f"Segmentation Masks ({self.method})")
            self.visualize(result, "segments_images", f"Segmentation Overlay ({self.method})")

        return result


def main():
    data_dir = os.path.abspath(r"img\btl4\GeometryFeature")
    image_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]

    pipeline = SegmentationPipeline(method="kmeans", kmeans_k=5, kmeans_use_position=True)
    pipeline.run(image_paths[:3], visualize=True)


if __name__ == "__main__":
    main()

