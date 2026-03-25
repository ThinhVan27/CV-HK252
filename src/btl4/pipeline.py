import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import wraps
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional, Tuple

def valid_input(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "input" in kwargs:
            input_data = kwargs["input"]
        elif len(args) >= 2:
            input_data = args[1]
        elif len(args) == 1:
            input_data = args[0]
        else:
            raise ValueError("Missing input argument")

        def valid_path(path):
            if not os.path.exists(path):
                raise ValueError(f"{path} do not exists")
            if not path.lower().endswith((".jpg", ".png", ".jpeg")):
                raise ValueError(f"{path} is not an image file")

        if isinstance(input_data, str):
            valid_path(input_data)
        elif isinstance(input_data, np.ndarray):
            pass
        elif isinstance(input_data, list):
            if not input_data:
                raise ValueError("Input list is empty")
            if all(isinstance(item, str) for item in input_data):
                for path in input_data:
                    valid_path(path)
            elif all(isinstance(item, np.ndarray) for item in input_data):
                pass
            else:
                raise TypeError("List input must be list[str] or list[np.ndarray]")
        else:
            raise TypeError("Input must be str, np.ndarray, list[str] or list[np.ndarray]")

        return func(*args, **kwargs)
    return wrapper
            
            
class BasePipeline(ABC):
    """
    Base Pipeline.
    """
    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Run pipeline.
        
        Returns:
          @result
        """
        pass

    def rgb2gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.copy()
        if image.ndim != 3:
            raise ValueError("Input image must have 2 or 3 dimensions")
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        raise ValueError("Unsupported channel size for grayscale conversion")

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim != 3:
            raise ValueError("Input image must have 2 or 3 dimensions")
        if image.shape[2] == 3:
            return image
        raise ValueError("Unsupported channel size")
    
    def _read_input(
        self, input: Union[str, List[str], List[np.ndarray], np.ndarray]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(input, str):
            image = cv2.imread(input, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read image from path: {input}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if isinstance(input, np.ndarray):
            return [self._ensure_rgb(input)]

        if isinstance(input, list):
            if all(isinstance(item, str) for item in input):
                images = []
                for path in input:
                    image = cv2.imread(path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Cannot read image from path: {path}")
                    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                return images

            if all(isinstance(item, np.ndarray) for item in input):
                return [self._ensure_rgb(image) for image in input]

        raise TypeError("Unsupported input type")
    

class DataPreprocessorPipeline(BasePipeline):
    """
    Image Preprocessing Pipeline
    """
    def __init__(
        self,
        resize_to: Optional[Tuple[int, int]] = None,
        crop_to: Optional[Tuple[int, int]] = None,
        apply_sharpen: bool = False,
        gaussian_ksize: int = 5,
        gaussian_sigma: float = 0.0,
    ):
        self.resize_to = resize_to
        self.crop_to = crop_to
        self.apply_sharpen = apply_sharpen
        self.gaussian_ksize = gaussian_ksize
        self.gaussian_sigma = gaussian_sigma

    @valid_input
    def run(self, input: Union[str, List[str], List[np.ndarray], np.ndarray]) -> Dict[str, Any]:
        """
        Run DataPreprocessorPipeline.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        images = self._read_input(input)

        processed_rgb = []
        processed_gray = []

        for image in images:
            current = image
            current = self._resize(current)
            current = self._crop(current)
            current = self._sharpening(current)

            processed_rgb.append(current)
            processed_gray.append(self.rgb2gray(current))

        result = {
            "rgb_images": processed_rgb,
            "gray_images": processed_gray,
            "num_images": len(processed_rgb),
            "settings": {
                "resize_to": self.resize_to,
                "crop_to": self.crop_to,
                "apply_sharpen": self.apply_sharpen,
                "gaussian_ksize": self.gaussian_ksize,
                "gaussian_sigma": self.gaussian_sigma,
            },
        }

        return result

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if self.resize_to is None:
            return image
        width, height = self.resize_to
        if width <= 0 or height <= 0:
            raise ValueError("resize_to must contain positive values")
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def _crop(self, image: np.ndarray) -> np.ndarray:
        if self.crop_to is None:
            return image

        crop_w, crop_h = self.crop_to
        if crop_w <= 0 or crop_h <= 0:
            raise ValueError("crop_to must contain positive values")

        h, w = image.shape[:2]
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        return image[y1:y2, x1:x2]

    def _sharpening(self, image: np.ndarray) -> np.ndarray:
        if not self.apply_sharpen:
            return image

        ksize = self.gaussian_ksize
        if ksize <= 0:
            raise ValueError("gaussian_ksize must be a positive odd integer")
        if ksize % 2 == 0:
            ksize += 1

        smooth_image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=self.gaussian_sigma)
        edges = cv2.subtract(image, smooth_image)
        sharpened = cv2.add(image, edges)
        
        return np.clip(sharpened, 0, 255)


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


class PanoramaStitchingPipeline(BasePipeline):
    """
    Panorama Image Stitching Pipeline.
    """
    
    @valid_input
    def run(self, input: Union[str, List[str], List[np.ndarray]]) -> Dict[str, Any]:
        """
        Run PanoramaStitchingPipeline.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        pass


class ObjectDetectionPipeline(BasePipeline):
    """
    Object Detection Pipeline
    """
    def __init__(self):
        pass

    @valid_input
    def run(self, input: Union[str, List[str], List[np.ndarray]]) -> Dict[str, Any]:
        """
        Run `OjectDetectionPipeline`.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        pass
    
    
class SegmentationPipeline(BasePipeline):
    """
    Segmentation Pipeline
    """
    
    @valid_input
    def run(self, input: Union[str, List[str], List[np.ndarray]]) -> Dict[str, Any]:
        """
        Run `SegmentationPipeline`.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        pass


class VisualizationPipeline(BasePipeline):
    """
    Visualization Pipeline.
    """
    def __init__(
        self,
        default_cols: int = 3,
        max_images_per_pipeline: int = 12,
        metadata_char_limit: int = 800,
    ):
        self.default_cols = max(1, default_cols)
        self.max_images_per_pipeline = max(1, max_images_per_pipeline)
        self.metadata_char_limit = max(100, metadata_char_limit)

    def run(
        self,
        results: Dict[str, Dict[str, Any]],
        cols: Optional[int] = None,
        max_images_per_pipeline: Optional[int] = None,
        assume_bgr: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(results, dict) or not results:
            raise ValueError("results must be a non-empty dict")

        effective_cols = max(1, cols or self.default_cols)
        effective_max_images = max(1, max_images_per_pipeline or self.max_images_per_pipeline)

        summaries: Dict[str, Dict[str, Any]] = {}

        for pipeline_name, pipeline_result in results.items():
            images = self._collect_images(pipeline_result)
            metadata = self._collect_metadata(pipeline_result)

            shown_images = images[:effective_max_images]
            self._plot_pipeline(
                name=pipeline_name,
                images=shown_images,
                metadata=metadata,
                cols=effective_cols,
                assume_bgr=assume_bgr,
            )

            summaries[pipeline_name] = {
                "num_images_found": len(images),
                "num_images_shown": len(shown_images),
                "metadata_keys": list(metadata.keys()),
            }

        plt.show()
        return summaries

    def _plot_pipeline(
        self,
        name: str,
        images: List[Tuple[str, np.ndarray]],
        metadata: Dict[str, Any],
        cols: int,
        assume_bgr: bool,
    ) -> None:
        n_images = len(images)
        n_meta_panel = 1 if metadata else 0
        total_panels = max(1, n_images + n_meta_panel)
        rows = (total_panels + cols - 1) // cols

        fig_w = 5 * cols
        fig_h = 4 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        axes = np.array(axes).reshape(-1)

        for idx in range(n_images):
            label, image = images[idx]
            ax = axes[idx]
            image_to_show = self._prepare_image_for_display(image, assume_bgr=assume_bgr)

            if image_to_show.ndim == 2:
                ax.imshow(image_to_show, cmap="gray")
            else:
                ax.imshow(image_to_show)

            ax.set_title(label, fontsize=10)
            ax.axis("off")

        if metadata:
            meta_ax = axes[n_images]
            meta_ax.axis("off")
            meta_ax.set_title("Metadata", fontsize=10)
            meta_text = self._stringify_metadata(metadata)
            meta_ax.text(0.0, 1.0, meta_text, va="top", ha="left", fontsize=9, wrap=True)

        for idx in range(n_images + n_meta_panel, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(f"{name} - Results Visualization", fontsize=13)
        fig.tight_layout()

    def _collect_images(self, data: Any, prefix: str = "") -> List[Tuple[str, np.ndarray]]:
        images: List[Tuple[str, np.ndarray]] = []

        if self._is_image_like(data):
            label = prefix or "image"
            images.append((label, data))
            return images

        if isinstance(data, dict):
            for key, value in data.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                images.extend(self._collect_images(value, next_prefix))
            return images

        if isinstance(data, (list, tuple)):
            for idx, value in enumerate(data):
                next_prefix = f"{prefix}[{idx}]" if prefix else f"item[{idx}]"
                images.extend(self._collect_images(value, next_prefix))
            return images

        return images

    def _collect_metadata(self, data: Any, prefix: str = "") -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        if self._is_image_like(data):
            return metadata

        if isinstance(data, dict):
            for key, value in data.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                nested = self._collect_metadata(value, next_prefix)
                metadata.update(nested)
            return metadata

        if isinstance(data, (list, tuple)):
            if data and any(self._is_image_like(item) for item in data):
                return metadata
            for idx, value in enumerate(data):
                next_prefix = f"{prefix}[{idx}]" if prefix else f"item[{idx}]"
                nested = self._collect_metadata(value, next_prefix)
                metadata.update(nested)
            return metadata

        key = prefix or "value"
        if isinstance(data, np.generic):
            metadata[key] = data.item()
        else:
            metadata[key] = data
        return metadata

    def _is_image_like(self, data: Any) -> bool:
        if isinstance(data, np.ndarray):
            return self._is_valid_image_shape(data)
        return False

    def _is_valid_image_shape(self, image: np.ndarray) -> bool:
        if image.ndim == 2:
            return True
        if image.ndim == 3 and image.shape[2] in (1, 3):
            return True
        return False

    def _prepare_image_for_display(self, image: np.ndarray, assume_bgr: bool = False) -> np.ndarray:
        out = np.asarray(image)

        if out.ndim == 3 and out.shape[2] == 1:
            out = out[:, :, 0]

        if assume_bgr and out.ndim == 3 and out.shape[2] == 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        return out

    def _stringify_metadata(self, metadata: Dict[str, Any]) -> str:
        if not metadata:
            return "No metadata"

        lines: List[str] = []
        for key, value in metadata.items():
            text = repr(value)
            if len(text) > 120:
                text = text[:117] + "..."
            lines.append(f"- {key}: {text}")

        combined = "\n".join(lines)
        if len(combined) > self.metadata_char_limit:
            return combined[: self.metadata_char_limit - 3] + "..."
        return combined


class OverallSceneAnalysisPipeline(BasePipeline):
    """
    Scence Analysis Pipeline
    """
    def __init__(self):
        # Composition của các pipeline con
        self.preprocessor = DataPreprocessorPipeline()
        self.geometry_1 = GeometryFeaturePipeline()
        self.geometry_2 = PanoramaStitchingPipeline()
        self.detection = ObjectDetectionPipeline()
        self.segmentation = SegmentationPipeline()
        self.viz = VisualizationPipeline()

        self.last_results = {}

    def run(self, input: Union[str, List[str], List[np.ndarray]], visualize=True) -> Dict[str, Dict[str, Any]]:
        """
        Run all pipeline and return each pipeline's result.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        res = {}
        # TODO
        # res['preprocess'] = self.preprocessor.run(input)
        # res['geometry_1'] = self.geometry_1.run(input)
        # res['geometry_2'] = self.geometry_2.run(input)
        # res['detection'] = self.detection.run(input)
        # res['segmentation'] = self.segmentation.run(input)
        
        self.last_results = res
        if visualize:
            self.visualize_all()
        return self.last_results

    def visualize_all(self):
        if self.last_results:
            self.viz.run(self.last_results)
        else:
            print("Have no results to visualize.")       