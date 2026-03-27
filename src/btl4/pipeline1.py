import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Optional, Tuple

from .pipeline import *

class GeometryFeaturePipeline(BasePipeline):
    """
    Boundaries, corners and edges detection pipeline.
    """
    def __init__(
        self,
        canny_threshold1: int = 100,
        canny_threshold2: int = 200,
        hough_threshold: int = 80,
        min_line_length: int = 30,
        max_line_gap: int = 10,
        max_corners: int = 200,
        corner_quality_level: float = 0.01,
        corner_min_distance: int = 8,
    ):
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.max_corners = max_corners
        self.corner_quality_level = corner_quality_level
        self.corner_min_distance = corner_min_distance

    
    @valid_input
    def run(self, input: Union[str, List[str], List[np.ndarray], np.ndarray], preprocess=None) -> Dict[str, Any]:
        """
        Run EdgeLineCornerPipeline.
        
        Args:
            @input: either image path, list of image paths or image ndarray.
            @preprocess: preprocessing result (optional)
        """
        rgb_images = self._read_input(input)

        gray_images: List[np.ndarray] = [self.rgb2gray(rgb) for rgb in rgb_images] if not preprocess else preprocess['gray_images']
        edges_images: List[np.ndarray] = []
        lines_images: List[np.ndarray] = []
        corners_images: List[np.ndarray] = []
        boundaries_images: List[np.ndarray] = []
        metrics: List[Dict[str, int]] = []

        for i, rgb in enumerate(rgb_images):
            gray = gray_images[i]
            
            edges = self._detect_edges(gray)
            lines_image, line_count = self._detect_lines(rgb, edges)
            corners_image, corner_count = self._detect_corners(rgb, gray)
            boundaries_image, boundary_count = self._detect_boundaries(rgb, edges)

            edges_images.append(edges)
            lines_images.append(lines_image)
            corners_images.append(corners_image)
            boundaries_images.append(boundaries_image)
            metrics.append(
                {
                    "num_lines": line_count,
                    "num_corners": corner_count,
                    "num_boundaries": boundary_count,
                }
            )

        result: Dict[str, Any] = {
            "rgb_images": rgb_images,
            "gray_images": gray_images,
            "edges_images": edges_images,
            "lines_images": lines_images,
            "corners_images": corners_images,
            "boundaries_images": boundaries_images,
            "metrics": metrics,
            "num_images": len(rgb_images),
            "settings": {
                "canny_threshold1": self.canny_threshold1,
                "canny_threshold2": self.canny_threshold2,
                "hough_threshold": self.hough_threshold,
                "min_line_length": self.min_line_length,
                "max_line_gap": self.max_line_gap,
                "max_corners": self.max_corners,
                "corner_quality_level": self.corner_quality_level,
                "corner_min_distance": self.corner_min_distance,
            },
        }

        return result

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        t1 = int(max(0, self.canny_threshold1))
        t2 = int(max(t1 + 1, self.canny_threshold2))
        return cv2.Canny(gray, threshold1=t1, threshold2=t2)

    def _detect_lines(self, rgb: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, int]:
        output = rgb.copy()
        threshold = max(1, int(self.hough_threshold))
        min_len = max(1, int(self.min_line_length))
        max_gap = max(0, int(self.max_line_gap))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )

        line_count = 0
        if lines is not None:
            for segment in lines:
                x1, y1, x2, y2 = segment[0]
                cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
                line_count += 1

        return output, line_count

    def _detect_corners(self, rgb: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, int]:
        output = rgb.copy()
        max_corners = max(1, int(self.max_corners))
        quality = float(max(1e-6, self.corner_quality_level))
        min_distance = max(1, int(self.corner_min_distance))

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality,
            minDistance=min_distance,
        )

        corner_count = 0
        if corners is not None:
            corners = corners.astype(np.int32)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(output, (x, y), 4, (0, 255, 0), 1)
                corner_count += 1

        return output, corner_count

    def _detect_boundaries(self, rgb: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, int]:
        output = rgb.copy()
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
        return output, len(contours)


