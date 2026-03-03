from enum import Enum
from dataclasses import dataclass
from typing import List, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt


class TransformationType(Enum):
    TRANSLATION = 1
    ROTATION = 2
    SCALING = 3
    SHEARING = 4
    AFFINE = 5
    PROJECTION = 6


type2str = {
    1: "Translation",
    2: "Rotation",
    3: "Scaling",
    4: "Shearing",
    5: "Affine",
    6: "Projection",
}


@dataclass
class Config:
    typ: TransformationType
    param_config: dict


class TransformationPipeline:
    def __init__(self, img_path, config: Union[List[Config], Config]):
        self.config = config
        self.img = self._prepare_img(img_path)

    @property
    def img_size(self):
        if self.img is not None:
            return (self.img.shape[0], self.img.shape[1])
        else:
            return None

    def _prepare_img(self, img_path):
        try:
            img = cv2.imread(img_path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def run(self):
        if isinstance(self.config, list):
            self._run_multiple_config()
        elif isinstance(self.config, Config):
            self._run_single_config(self.config)
        else:
            print("[ERROR] config type is error")

    def _run_single_config(self, config, visualize=True):
        if config.typ == TransformationType.TRANSLATION:
            tx = config.param_config["tx"]
            ty = config.param_config["ty"]
            matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        elif config.typ == TransformationType.SCALING:
            sx = config.param_config["sx"]
            sy = config.param_config["sy"]
            matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)
        elif config.typ == TransformationType.SHEARING:
            kx = config.param_config["kx"]
            ky = config.param_config["ky"]
            matrix = np.array([[1, kx, 0], [ky, 1, 0], [0, 0, 1]], dtype=np.float32)

        elif config.typ == TransformationType.ROTATION:
            cx = config.param_config["cx"]
            cy = config.param_config["cy"]
            angle = config.param_config["angle"]
            rotate_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1)
            matrix = np.eye(3, 3)
            matrix[:2, :] = rotate_matrix
        elif config.typ == TransformationType.AFFINE:
            matrix = np.eye(3, 3)
            matrix[:2, :] = cv2.getAffineTransform(**config.param_config)
        elif config.typ == TransformationType.PROJECTION:
            matrix = cv2.getPerspectiveTransform(**config.param_config)
        else:
            print("[ERROR] the property type of config is out of this feature")
            return
        result = cv2.warpPerspective(
            self.img, matrix, (self.img.shape[1], self.img.shape[0])
        )
        if visualize:
            self._visualize(result)
        return result

    def _run_multiple_config(self):
        results = [
            self._run_single_config(config, visualize=False) for config in self.config
        ]
        self._visualize(results)

    def _visualize(self, result):
        plt.imshow(self.img)
        plt.title("Origin")
        plt.axis("off")
        plt.show()
        if isinstance(result, list):
            num = len(result)
            n_row = int(np.ceil(num / 2))
            fig, ax = plt.subplots(n_row, 2, figsize=(5 * n_row, 10), squeeze=False)
            for i in range(int(np.ceil(num / 2))):
                for j in range(2):
                    if i * 2 + j < num and result[i * 2 + j] is not None:
                        ax[i][j].imshow(result[i * 2 + j])
                        ax[i][j].axis("off")
                        ax[i][j].set_title(type2str[self.config[i * 2 + j].typ.value])
                    else:
                        ax[i][j].axis("off")
            fig.tight_layout()
            plt.show()
        else:
            plt.imshow(result)
            plt.title(type2str[self.config.typ.value])
            plt.axis("off")
            plt.tight_layout()
            plt.show()
