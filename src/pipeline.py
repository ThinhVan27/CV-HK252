import os
from enum import Enum
from typing import List, TypedDict, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

class Config(TypedDict):
    """
    Configuration for the SpatialFilteringPipeline.

    Attributes:
        kernel: Convolution kernel for filtering.
        name: Name of kernel, if provided.
    """

    kernel: np.ndarray
    name: str


class FilterType(Enum):
    """Enumeration for filter types used in the SpatialFilteringPipeline."""

    HIGHPASS = 0
    LOWPASS = 1


class SpatialFilteringPipeline:
    """
    Spatial filtering (low-pass/high-pass) pipeline for image processing.

    This class applies filtering to images using a specified convolution kernel.
    """

    def __init__(self, config: Config, type: FilterType = FilterType.HIGHPASS):
        """
        Initialize the spatial filtering pipeline.

        Args:
            config: Configuration containing kernel and name.
            type: Type of filter (HIGHPASS or LOWPASS).
        """
        self.config = config
        self.type = type

    def run(self, img_path: Union[str, List[str]], keep_RGB: bool = True):
        """
        Run spatial filtering on image(s).

        Args:
            img_path: Path to a single image or list of image paths.
            keep_RGB: If True (LOWPASS only), keep RGB color; if False, convert to grayscale.
        """
        if isinstance(img_path, str):
            if not os.path.exists(img_path):
                print(f"Invalid Path {img_path}")
                return
            color_mode = (
                1 if (self.type == FilterType.LOWPASS and keep_RGB) else 0
            )  # Chế độ màu
            img_origin = cv2.imread(
                img_path, color_mode
            )  # Load ảnh và chuyển sang chế độ màu mong muốn
            # Đổi từ BGR -> RGB nếu ảnh ở định dạng 3 kênh
            if color_mode == 1:
                img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
            img = cv2.filter2D(
                img_origin, cv2.CV_32F, self.config["kernel"]
            )  # Thực hiện filter trên miền không gian

            # Hiện thực hậu xử lí
            if self.type == FilterType.HIGHPASS:
                img = np.abs(img)  # Giữ biên âm nếu High-Pass
            img = np.clip(img, 0, 255).astype(
                np.uint8
            )  # Chuyển giá trị pixel về [0, 255] và về kiểu uint8 chuẩn của ảnh
            # Hiển thị ảnh
            _, axis = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
            cmap_type = "gray" if color_mode == 0 else None
            axis[0].imshow(img_origin, cmap=cmap_type)
            axis[0].set_title("Origin", fontsize=20)
            axis[0].axis("off")
            axis[1].imshow(img, cmap=cmap_type)
            axis[1].set_title("Output", fontsize=20)
            axis[1].axis("off")
            plt.show()
        elif isinstance(img_path, list):
            for path in img_path:
                self.run(path, keep_RGB)
        else:
            print("Invalid type for parameter 'img_path'")


if __name__ == "__main__":
    kernel = 1 / 36 * np.ones((6, 6))
    config = Config(kernel=kernel, name="MedianBlur")
    pl = SpatialFilteringPipeline(config=config, type=FilterType.LOWPASS)
    pl.run(["img/lion.jpg", "img/sky.jpg"], keep_RGB=True)
