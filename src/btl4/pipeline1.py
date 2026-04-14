import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Optional, Tuple, Literal
from abc import ABC, abstractmethod

from base_pipeline import *


class Detector(ABC):
    """Base interface for all geometry feature detectors."""

    @abstractmethod
    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError


class EdgeDetector(Detector):
    """Interface for edge detection algorithms."""


class LineDetector(Detector):
    """Hough transform based detector for line features."""

    def __init__(self, 
                 threshold: int = 100, 
                 min_line_length: int = 80, 
                 max_line_gap: int = 10):
        self.threshold = int(max(1, threshold))
        self.min_line_length = int(max(1, min_line_length))
        self.max_line_gap = int(max(0, max_line_gap))

    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        lines = cv2.HoughLinesP(
            input,
            rho=1,
            theta=np.pi / 180,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        return {
            "lines": [] if lines is None else lines,
            "count": 0 if lines is None else int(len(lines)),
        }


class CornerDetector(Detector):
    """Interface for corner detection algorithms."""
    pass

class SobelFilter(EdgeDetector):
    def __init__(self, 
                 ksize: int = 3, 
                 threshold: int = 20,
                 order: Literal["x", "y"] = "x"):
        if ksize <= 0:
            raise ValueError("ksize must be > 0")
        self.ksize = int(ksize if ksize % 2 == 1 else ksize + 1)
        self.threshold = int(max(0, threshold))
        self.dx = 1 if order == "x" else 0

    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        lap = cv2.Sobel(input, cv2.CV_64F, dx=self.dx, dy=1 - self.dx, ksize=self.ksize)
        lap_abs = cv2.convertScaleAbs(lap)
        _, edges = cv2.threshold(lap_abs, self.threshold, 255, cv2.THRESH_BINARY)
        return {
            "edges": edges.astype(np.uint8),
        }
        

class LaplacianFilter(EdgeDetector):
    def __init__(self, ksize: int = 3, threshold: int = 20):
        if ksize <= 0:
            raise ValueError("ksize must be > 0")
        self.ksize = int(ksize if ksize % 2 == 1 else ksize + 1)
        self.threshold = int(max(0, threshold))

    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        lap = cv2.Laplacian(input, cv2.CV_64F, ksize=self.ksize)
        lap_abs = cv2.convertScaleAbs(lap)
        _, edges = cv2.threshold(lap_abs, self.threshold, 255, cv2.THRESH_BINARY)
        return {
            "edges": edges.astype(np.uint8),
        }


class Canny(EdgeDetector):
    def __init__(self, threshold1: int = 50, threshold2: int = 150, aperture_size: int = 3):
        self.threshold1 = int(max(0, threshold1))
        self.threshold2 = int(max(self.threshold1 + 1, threshold2))
        if aperture_size not in (3, 5, 7):
            aperture_size = 3
        self.aperture_size = int(aperture_size)

    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        edges = cv2.Canny(
            input,
            threshold1=self.threshold1,
            threshold2=self.threshold2,
            apertureSize=self.aperture_size,
            L2gradient=True,
        )
        return {
            "edges": edges.astype(np.uint8),
        }


class HarrisAlgo(CornerDetector):
    def __init__(self, 
                 block_size: int = 2, 
                 ksize: int = 3,
                 k: float = 0.04, 
                 threshold_ratio: float = 0.01):
        self.block_size = int(max(2, block_size))
        self.ksize = int(ksize if ksize % 2 == 1 else ksize + 1)
        self.k = float(k)
        self.threshold_ratio = float(max(1e-5, threshold_ratio))

    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        gray_f = np.float32(input)
        response = cv2.cornerHarris(gray_f, self.block_size, self.ksize, self.k)
        response = cv2.dilate(response, None)
        threshold = self.threshold_ratio * float(response.max()) if response.size else 0.0
        points = np.argwhere(response > threshold)
        # np.argwhere returns [y, x], convert to [x, y]
        points_xy = np.array([[int(p[1]), int(p[0])] for p in points], dtype=np.int32)
        return {
            "points": points_xy,
            "count": int(points_xy.shape[0]),
            "response": response,
        }


class ShiTomasiAlgo(CornerDetector):
    def __init__(self, max_corners: int = 200, quality_level: float = 0.01, min_distance: float = 8.0):
        self.max_corners = int(max(1, max_corners))
        self.quality_level = float(max(1e-5, quality_level))
        self.min_distance = float(max(0.1, min_distance))

    def extract(self, input: np.ndarray) -> Dict[str, Any]:
        corners = cv2.goodFeaturesToTrack(
            input,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level, # minimum eigenvalue for a corner
            minDistance=self.min_distance, # minimum distance between any corners
            useHarrisDetector=False,
        )
        if corners is None:
            points = np.zeros((0, 2), dtype=np.int32)
        else:
            points = corners.reshape(-1, 2).astype(np.int32)

        return {
            "points": points,
            "count": int(points.shape[0]),
        }


class GeometryFeaturePipeline(BasePipeline):
    """
    Boundaries, corners and edges detection pipeline.
    """

    def __init__(
        self,
        edge_detector: EdgeDetector = Canny(),
        corner_detector: CornerDetector = HarrisAlgo(),
        line_detector: LineDetector = LineDetector(),
        random_seed: int = 42
    ):
        self.random_seed = int(random_seed)
        np.random.seed(self.random_seed)
        cv2.setRNGSeed(self.random_seed)
        self.edge_detector = edge_detector
        self.corner_detector = corner_detector
        self.line_detector = line_detector

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        return self.edge_detector.extract(gray)["edges"]

    def _detect_lines(self, rgb: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, int]:
        out = rgb.copy()
        detected = self.line_detector.extract(edges)
        lines = detected["lines"]
        count = int(detected["count"])

        if count > 0:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return out, count

    def _detect_corners(self, rgb: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, int]:
        out = rgb.copy()
        detected = self.corner_detector.extract(gray)
        points = detected["points"]
        count = int(detected["count"])

        if count > 0:
            for x, y in points:
                cv2.circle(out, (int(x), int(y)), 5, (0, 255, 0), 3)

        return out, count
    
    @valid_input
    def run(self, input: Union[str, List[str], List[np.ndarray], np.ndarray]) -> Dict[str, Any]:
        """
        Run EdgeLineCornerPipeline.
        
        Args:
            @input: either image path, list of image paths or image ndarray.
            @preprocess: preprocessing result (optional)
        """
        rgb_images = self._read_input(input)

        gray_images: List[np.ndarray] = [self.rgb2gray(rgb) for rgb in rgb_images]
        edges_images: List[np.ndarray] = []
        lines_images: List[np.ndarray] = []
        corners_images: List[np.ndarray] = []
        metrics: List[Dict[str, int]] = []

        for i, rgb in enumerate(rgb_images):
            gray = gray_images[i]
            
            edges = self._detect_edges(gray)
            lines_image, line_count = self._detect_lines(rgb, edges)
            corners_image, corner_count = self._detect_corners(rgb, gray)

            edges_images.append(edges)
            lines_images.append(lines_image)
            corners_images.append(corners_image)
            metrics.append(
                {
                    "num_lines": line_count,
                    "num_corners": corner_count,
                }
            )

        result: Dict[str, Any] = {
            "rgb_images": rgb_images,
            "gray_images": gray_images,
            "edges_images": edges_images,
            "lines_images": lines_images,
            "corners_images": corners_images,
            "metrics": metrics,
            "num_images": len(rgb_images),
        }
        self.visualize(result, "corners_images", "Corner Detection")
        self.visualize(result, "lines_images", "Line Detection")
        self.visualize(result, "edges_images", "Edge Detection")
        return result
        
def main():
    pipeline = GeometryFeaturePipeline(edge_detector=Canny(threshold1=50, threshold2=150),
                                       corner_detector=ShiTomasiAlgo(),
                                       line_detector=LineDetector(200, 130, 10))
    data_dir = os.path.abspath(r"img\btl4\GeometryFeature")
    pipeline.run([os.path.join(data_dir, dir) for dir in os.listdir(data_dir)])
    
if __name__ == "__main__":
    main()
