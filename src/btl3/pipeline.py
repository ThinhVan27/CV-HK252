from abc import ABC, abstractmethod
import os
import cv2
from typing import List, TypedDict
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


# ======================================
# Định nghĩa các bộ trích xuất đặc trưng
# ======================================
class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, img: np.ndarray):
        """
        return:
            keypoint và feature(descriptor)
        """
        pass


class SIFT(FeatureExtractor):
    def __init__(self):
        self.al = cv2.SIFT_create()

    def extract(self, img):
        return self.al.detectAndCompute(img, None)


class ORB(FeatureExtractor):
    def __init__(self):
        self.al = cv2.ORB_create()

    def extract(self, img):
        return self.al.detectAndCompute(img, None)


# TODO: thêm các bộ trích xuất đặc trưng
# Yêu cầu phải kế thừa FeatureExtractor
# ................


# =======================================
# Trộn ảnh
# =======================================
class BlendingBase(ABC):
    @abstractmethod
    def blend(self, query_img, train_img, homo_matrix):
        pass


class AlphaBlending(BlendingBase):

    def blend(self, query_image, train_image, homo_matrix):
        width_query_img = query_image.shape[1]
        width_train_img = train_image.shape[1]
        lowest_width = min(width_query_img, width_train_img)
        smoothing_window_percent = 0.10
        self.smoothing_window_size = max(
            100, min(smoothing_window_percent * lowest_width, 1000)
        )
        height_img1 = query_image.shape[0]
        width_img1 = query_image.shape[1]
        width_img2 = train_image.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3), dtype=np.float32)
        mask1 = self._create_mask(query_image, train_image, version="left")
        panorama1[0 : query_image.shape[0], 0 : query_image.shape[1], :] = (
            query_image.astype(np.float32)
        )
        panorama1 *= mask1
        mask2 = self._create_mask(query_image, train_image, version="right")
        panorama2 = (
            cv2.warpPerspective(
                train_image, homo_matrix, (width_panorama, height_panorama)
            )
            * mask2
        )
        result = panorama1 + panorama2
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return np.clip(final_result, 0, 255).astype(np.uint8)

    def _create_mask(self, query_image, train_image, version="left"):
        height_query_img = query_image.shape[0]
        width_query_img = query_image.shape[1]
        width_train_img = train_image.shape[1]
        height_panorama = height_query_img
        width_panorama = width_query_img + width_train_img
        offset = int(self.smoothing_window_size / 2)
        barrier = query_image.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama), dtype=np.float32)
        if version == "left":
            mask[:, barrier - offset : barrier + offset] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, barrier + offset :] = 1
        return cv2.merge([mask, mask, mask])


# TODO: thêm các thuật toán trộn ảnh
# Yêu cầu phải kế thừa Blending


class Config(TypedDict):
    extractor: FeatureExtractor
    blending: BlendingBase


class Pipeline:
    def __init__(self, config: Config):
        self.extractor = config["extractor"]
        self.blending = config["blending"]

    def run(self, img_root, image_names: List[str]):
        """
        Run pipeline to visualize the target images after stitching
        Args:
            img_root: Đường dẫn thư mục chứa ảnh
            image_names: Tên các ảnh cần ghép từ trái sang phải
        """
        images = [cv2.imread(os.path.join(img_root, e)) for e in image_names]
        try:
            if len(images) < 2:
                raise ValueError("At least 2 images are required for stitching")
            if len(images) == 2:
                result = self._run_two_image(images[0], images[1])
            else:
                result = self._run_multiple_image(images)
            Pipeline.visualize(result)
        except Exception as e:
            print(f"[Run]: {e}")

    def _preprocessing(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img, img

    def _post_processing(self, x):
        return x

    def _run_two_image(self, left_img, right_img):
        # ======================================
        # Stage 1: Preprocessing: Chuyển về ảnh xám
        # ======================================
        print("_run_two_image")
        gray_train_img, train_img = self._preprocessing(left_img)
        print("_run_two_image")
        gray_query_img, query_img = self._preprocessing(right_img)
        print("_run_two_image")
        # ======================================
        # Stage 2: Trích xuất đặc trưng
        # ======================================
        keypoints_train, features_train = self.extractor.extract(gray_train_img)
        keypoints_query, features_query = self.extractor.extract(gray_query_img)
        # ======================================
        # Stage 3: Matching các đặc trưng
        # ======================================
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        best_matches = bf.match(features_query, features_train)
        matches = sorted(best_matches, key=lambda x: x.distance)

        # ======================================
        # Stage 4: Tìm ma trận Homo
        # ======================================
        _, H, _ = Pipeline.compute_homography(
            keypoints_train, keypoints_query, matches, 5
        )

        # ======================================
        # Stage 5: Căn chỉnh, ghép và trộn ảnh
        # ======================================
        # H maps left(train) -> right(query), so invert it to warp right -> left.
        H_inv = np.linalg.inv(H)
        result = self.blending.blend(train_img, query_img, H_inv)
        # BGR
        # ======================================
        # Stage 6: Hậu xử lí
        # ======================================
        final_result = self._post_processing(result)
        print("finale")
        return final_result

    def _run_multiple_image(self, images):
        print("multiple")
        return reduce(lambda x, y: self._run_two_image(x, y), images[1:], images[0])

    @staticmethod
    def compute_homography(keypoints_train, keypoints_query, matches, reprojThresh):
        keypoints_train = np.float32([keypoint.pt for keypoint in keypoints_train])
        keypoints_query = np.float32([keypoint.pt for keypoint in keypoints_query])

        if len(matches) >= 4:
            points_train = np.float32([keypoints_train[m.trainIdx] for m in matches])
            points_query = np.float32([keypoints_query[m.queryIdx] for m in matches])

            H, status = cv2.findHomography(
                points_train, points_query, cv2.RANSAC, reprojThresh
            )

            return (matches, H, status)

        else:
            raise RuntimeError(
                "Minimum match count not satisfied cannot get homopgrahy"
            )

    @staticmethod
    def visualize(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title("Result of Image Stitching")
        plt.show()


if __name__ == "__main__":

    config = {"extractor": SIFT(), "blending": AlphaBlending()}
    pl = Pipeline(config)
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    root = os.path.join(root, "img", "btl3")
    pl.run(root, ["image1.jpg", "image2.jpg", "image3.jpg"])
