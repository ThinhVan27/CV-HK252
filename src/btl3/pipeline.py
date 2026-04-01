from abc import ABC, abstractmethod
import os
import cv2
from typing import List, TypedDict
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from evaluation import *
import pandas as pd

# ======================================
# Định nghĩa các bộ trích xuất đặc trưng
# ======================================
class FeatureExtractor(ABC):
    matcher_norm = cv2.NORM_L2

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
    
    def __str__(self):
        return "SIFT"


class ORB(FeatureExtractor):
    matcher_norm = cv2.NORM_HAMMING

    def __init__(self):
        self.al = cv2.ORB_create()

    def extract(self, img):
        return self.al.detectAndCompute(img, None)
    
    def __str__(self):
        return "ORB"


class PCASIFT(FeatureExtractor):
    """PCA-SIFT approximation: detect with SIFT, then reduce descriptors by PCA."""

    matcher_norm = cv2.NORM_L2

    def __init__(self, n_components: int = 36, nfeatures: int = 0):
        self.al = cv2.SIFT_create(nfeatures=nfeatures)
        self.n_components = n_components

    def _pca_reduce(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors.ndim != 2:
            raise ValueError("Descriptors must have shape (N, D).")

        num_components = int(
            max(1, min(self.n_components, descriptors.shape[0], descriptors.shape[1]))
        )
        centered = descriptors - np.mean(descriptors, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projection = vt[:num_components]
        reduced = centered @ projection.T

        # Normalize each descriptor for more stable matching with L2 distance.
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        reduced = reduced / np.clip(norms, 1e-12, None)
        return reduced.astype(np.float32)

    def extract(self, img):
        keypoints, descriptors = self.al.detectAndCompute(img, None)
        if descriptors is None:
            return keypoints, None
        descriptors = descriptors.astype(np.float32)
        descriptors = self._pca_reduce(descriptors)
        return keypoints, descriptors

    def __str__(self):
        return "PCA-SIFT"

class AKAZE(FeatureExtractor):
    matcher_norm = cv2.NORM_HAMMING

    def __init__(self):
        self.al = cv2.AKAZE_create()

    def extract(self, img):
        return self.al.detectAndCompute(img, None)
    
    def __str__(self):
        return "AKAZE"

# =======================================
# Trộn ảnh
# =======================================
class BlendingBase(ABC):

    @abstractmethod
    def blend(self, train_img, query_img, homo_matrix):
        pass
    
    def _post_processing(self, result):
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return np.clip(final_result, 0, 255).astype(np.uint8)


class AlphaBlending(BlendingBase):
    
    def accept(self, visitor):
        return visitor.visitAlpha(self)
    
    def blend(self, train_image, query_image, homo_matrix):
        # return self.blend_(train_image, query_image, homo_matrix)
        height_img1 = train_image.shape[0]
        width_img1 = train_image.shape[1]
        width_img2 = query_image.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        panorama1 = np.zeros((height_panorama, width_panorama, 3), dtype=np.float32)
        mask1 = self._create_mask(train_image, query_image, version="left")
        panorama1[:, : width_img1, :] = (
            train_image.astype(np.float32)
        )
        cache_panorama1 = panorama1
        panorama1 *= mask1
        mask2 = self._create_mask(train_image, query_image, version="right")
        panorama2 = cv2.warpPerspective(
                query_image, homo_matrix, (width_panorama, height_panorama)
            ).astype(np.float32)
        cache_panorama2 = panorama2
        panorama2 *= mask2
        self.panorama = (cache_panorama1, cache_panorama2)
        result = panorama1 + panorama2
        return self._post_processing(result)

    def _create_mask(self, train_image, query_image, version="left"):
        height_train_img, width_train_img = train_image.shape[:2]
        width_query_img = query_image.shape[1]
        height_panorama = height_train_img
        width_panorama = width_query_img + width_train_img
        smoothing_window_percent = 0.3
        smoothing_window_size = max(
            100, min(smoothing_window_percent * min(width_query_img, width_train_img), 1000)
        )
        offset = int(smoothing_window_size / 2)
        barrier = train_image.shape[1] - offset
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

    def __str__(self):
        return "Alpha Blending"

class PoissonBlending(BlendingBase):
    """Poisson blending class"""
    
    def __init__(self):
        super().__init__()
        
    def accept(self, visitor):
        return visitor.visitPoisson(self)
    
    def blend(self, train_img, query_img, H):
        """Blending `query_img` on the right of `train_img`

        Args:
            train_img (`np.ndarray`)
            query_img (`np.ndarray`)
            H (`np.ndarray`): _homography matrix to transform_ `query_img` _to_ `train_img`

        Returns:
            result: _blended image_
        """
        panorama1, panorama2 = self._init_panorama(train_img, query_img, H)
        blended_roi, result = self._blend_roi_region(panorama1, panorama2)
        return self._post_processing(result)
    
    def _init_panorama(self, train_img, query_img, H):
        """ Initialize panorama base.

        Args:
            train_img (`np.ndarray`):
            query_img (`np.ndarray`):
            H (`np.ndarray`):  _homography matrix to transform_ `query_img` _to_ `train_img`

        Returns:
            panorama1, panorama2 
        """
        height_train, width_train = train_img.shape[:2]
        height_query, width_query = query_img.shape[:2]
        height_panorama = height_train
        width_panorama = width_query + width_train
        panorama1 = np.zeros((height_panorama, width_panorama, 3), dtype=np.uint8)
        panorama1[:height_train, :width_train, :] = train_img
        panorama2 = cv2.warpPerspective(query_img, H, (width_panorama, height_panorama)).astype(np.uint8)
        self.panorama = (panorama1, panorama2)
        return panorama1, panorama2
    
    def _blend_roi_region(self, panorama1, panorama2):
        """Blending ROI region

        Args:
            panorama1 (`np.ndarray`)
            panorama2 (`np.ndarray`)

        Returns:
            roi_region: _blended region_
            result: _blended image_
        """
        mask1 = np.any(panorama1 != 0, axis=-1)
        base_target = panorama2.copy()
        base_target[mask1] = panorama1[mask1] 
        mask2 = np.any(panorama2 != 0, axis=-1).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask2_eroded = cv2.erode(mask2, kernel, iterations=1)
        x, y, w, h = cv2.boundingRect(mask2_eroded)
        if w == 0 or h == 0:
            return self._post_processing(base_target)
        roi_src = panorama2[y:y+h, x:x+w]
        roi_dst = base_target[y:y+h, x:x+w]
        roi_mask = mask2_eroded[y:y+h, x:x+w]
        center = (w // 2, h // 2)
        try:
            blended_roi = cv2.seamlessClone(
                src=roi_src,
                dst=roi_dst,
                mask=roi_mask,
                p=center,
                flags=cv2.NORMAL_CLONE
            )
        except Exception as e:
            print(f"[ERROR] Error at bleding process: {repr(e)}")
            blended_roi = roi_dst
        blended_img = base_target.copy()
        blended_img[y:y+h, x:x+w] = blended_roi
        return blended_roi, blended_img
        
    def __str__(self):
        return "Poisson Blending"
    
        
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
            Pipeline.visualize(result['result'])
            return result
        except Exception as e:
            raise e
            # print(f"[Run]: {e}")

    def _preprocessing(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img, img

    def _post_processing(self, x):
        return x

    def _match_features(self, features_query, features_train):
        # FLANN needs LSH for binary descriptors and KDTree for float descriptors.
        if features_query.dtype == np.uint8 and features_train.dtype == np.uint8:
            index_params = {
                "algorithm": 6,
                "table_number": 6,
                "key_size": 12,
                "multi_probe_level": 1,
            }
            search_params = {"checks": 50}
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            knn_matches = matcher.knnMatch(features_query, features_train, k=2)
        else:
            query_float = np.asarray(features_query, dtype=np.float32)
            train_float = np.asarray(features_train, dtype=np.float32)
            index_params = {"algorithm": 1, "trees": 5}
            search_params = {"checks": 50}
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            knn_matches = matcher.knnMatch(query_float, train_float, k=2)

        ratio_thresh = 0.75
        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        return sorted(good_matches, key=lambda x: x.distance)

    def _run_two_image(self, left_img, right_img):
        # ======================================
        # Stage 1: Preprocessing: Chuyển về ảnh xám
        # ======================================
        gray_train_img, train_img = self._preprocessing(left_img)
        gray_query_img, query_img = self._preprocessing(right_img)
        # ======================================
        # Stage 2: Trích xuất đặc trưng
        # ======================================
        keypoints_train, features_train = self.extractor.extract(gray_train_img)
        keypoints_query, features_query = self.extractor.extract(gray_query_img)
        # ======================================
        # Stage 3: Matching các đặc trưng
        # ======================================
        if features_train is None or features_query is None:
            raise RuntimeError("Cannot match features because descriptor extraction failed.")

        matches = self._match_features(features_query, features_train)
        if len(matches) < 4:
            raise RuntimeError(
                f"Not enough good matches after FLANN ratio test: {len(matches)}"
            )

        # ======================================
        # Stage 4: Tìm ma trận Homo
        # ======================================
        _, H, _ = Pipeline.compute_homography(
            keypoints_train, keypoints_query, matches, 5
        )

        # ======================================
        # Stage 5: Căn chỉnh, ghép và trộn ảnh
        # ======================================
        H_inv = np.linalg.inv(H)
        result_img = self.blending.blend(train_img, query_img, H_inv)
        # BGR
        # ======================================
        # Stage 6: Hậu xử lí
        # ======================================
        result_img = self._post_processing(result_img)
        
        info = {
            "result": result_img,
            "kp_train": len(keypoints_train),
            "kp_query": len(keypoints_query),
            "n_matches": len(matches)
        }
        return info

    def _run_multiple_image(self, images):
        mid = len(images) // 2
        left_images = images[:mid]
        right_images = images[mid:]
        left_result = reduce(lambda x, y: self._run_two_image(y, x)['result'], left_images[1:], left_images[0])
        result = reduce(lambda x, y: self._run_two_image(x, y)['result'], right_images, left_result)
        return {"result": result}

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
    def visualize(image, name=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.axis(False)
        plt.tight_layout()
        plt.imshow(image)
        plt.title("Result of Image Stitching" if not name else name)
        plt.show()


if __name__ == "__main__":

    v = BlendingVisitor()
    blending = PoissonBlending()
    config = {"extractor": AKAZE(), "blending": blending}
    pl = Pipeline(config)
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    root = os.path.join(root, "img", "btl3")
    result = pl.run(os.path.join(root, "Lab"), [ "image2.jpg","image3.jpg"])
    print(f"Num keypoints left: {result.get("kp_train", 0)} | Num keypoints right: {result.get("kp_query", 0)} | Num matches: {result.get("n_matches", 0)}")
    print(blending.accept(v))