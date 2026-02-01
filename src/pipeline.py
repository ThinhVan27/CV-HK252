from typing import Dict, Union, List, TypedDict
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Config(TypedDict):
    """
    Configuration for the SpatialFilteringPipeline.

    Attributes:
        kernel: Convolution kernel for filtering.
        name: Name of kernel, if provided
    """

    kernel: np.ndarray
    name: str


class SpatialFilteringPipeline:
    """
    Spatial filtering (low-pass/high-pass) pipeline for image processing.

    This class applies filtering to images using a specified convolution kernel.
    """

    def __init__(self, config: Config, type="High-pass"):
        self.config = config
        self.type = type

    def run(self, img_path: Union[str, List[str]]):
        """
        Run Domain filtering on image(s).

        Args:
            img_path: Path to a single image or list of image paths.
        """
        if isinstance(img_path, str):
            img_origin = cv2.imread(
                img_path, 0
            )  # Load ảnh và chuyển sang dạng grayscale
            img = cv2.filter2D(
                img_origin, cv2.CV_32F, self.config["kernel"]
            )  # Thực hiện filter trên miền không gian
            img = np.abs(img)
            img = np.clip(img, 0, 255).astype(
                np.uint8
            )  # Chuyển về kiểu chuẩn của ảnh là uint8
            # Hiển thị ảnh
            fig, axis = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
            axis[0].imshow(img_origin, cmap="gray")
            axis[0].set_title("Origin", fontsize=20)
            axis[0].axis("off")
            axis[1].imshow(img, cmap="gray")
            axis[1].set_title("Output", fontsize=20)
            axis[1].axis("off")
        elif isinstance(img_path, list):
            for path in img_path:
                self.run(path)
        else:
            print("Invalid type for parameter 'img_path'")
            return None


if __name__ == "__main__":
    pass
