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
            return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]

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
    
    def visualize(self, result, what, name):
        n_imgs = len(result[what])
        cols = min(3, n_imgs)
        rows = (n_imgs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 4*rows))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < n_imgs:
                img = result[what][i]
                if len(img.shape) == 2:
                    ax.imshow(result[what][i], cmap="gray")
                else:
                    ax.imshow(result[what][i])
            axes[i].axis("off")
        fig.suptitle(name)
        plt.tight_layout()
        plt.show()
    
    
class DataPreprocessorPipeline(BasePipeline):
    """
    Image Preprocessing Pipeline
    """
    def __init__(
        self,
        resize_to: Optional[Tuple[int, int]] = None,
        crop_to: Optional[Tuple[int, int]] = None,
        apply_sharpen: bool = False,
        apply_smoothing: bool = False,
        gaussian_ksize: int = 5,
        gaussian_sigma: float = 0.0,
    ):
        self.resize_to = resize_to
        self.crop_to = crop_to
        self.apply_sharpen = apply_sharpen
        self.apply_smoothing = apply_smoothing
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
            if self.apply_sharpen:
                current = self._sharpening(current)
            if self.apply_smoothing:
                current = self._smoothing(current)

            processed_rgb.append(current)
            processed_gray.append(self.rgb2gray(current))

        result = {
            "rgb_images": processed_rgb,
            "gray_images": processed_gray,
            "num_images": len(processed_rgb),
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

    def _smoothing(self, image: np.ndarray) -> np.ndarray:
        if not self.apply_smoothing:
            return image
        ksize = self.gaussian_ksize
        if ksize <= 0:
            raise ValueError("gaussian_ksize must be a positive odd integer")
        if ksize % 2 == 0:
            ksize += 1
        smooth_img = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=self.gaussian_sigma)
        return np.clip(smooth_img, 0, 255).astype(np.uint8)
    
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
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)