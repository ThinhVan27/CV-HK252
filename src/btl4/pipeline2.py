import os

from typing import Dict, Any, List, Union, Optional, TypedDict

try:
    from .base_pipeline import BasePipeline, valid_input
except ImportError:
    from base_pipeline import BasePipeline, valid_input


# ==============
# ==============
# ==============


from abc import ABC, abstractmethod
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.btl3.evaluation import evaluate_from_matches
import pandas as pd
import time

try:
    import onnxruntime as ort
except Exception:
    ort = None


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


class AKAZE(FeatureExtractor):
    matcher_norm = cv2.NORM_HAMMING

    def __init__(self):
        self.al = cv2.AKAZE_create()

    def extract(self, img):
        return self.al.detectAndCompute(img, None)

    def __str__(self):
        return "AKAZE"


class SuperPointExtractor(FeatureExtractor):
    """SuperPoint feature extractor backed by an ONNX model."""

    matcher_norm = cv2.NORM_L2

    def __init__(
        self,
        max_keypoints: int = 1024,
        keypoint_threshold: float = 0.005,
        nms_radius: int = 4,
        remove_borders: int = 4,
    ):
        src_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(src_root, "checkpoint", "superpoint_no_borders.onnx")
        self.model_path = model_path
        self.max_keypoints = max(1, int(max_keypoints))
        self.keypoint_threshold = float(keypoint_threshold)
        self.nms_radius = max(0, int(nms_radius))
        self.remove_borders = max(0, int(remove_borders))
        self.net: Optional[cv2.dnn_Net] = None
        self.session = None

    def _load_net(self):
        if self.session is not None or self.net is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SuperPoint model not found: {self.model_path}")
        if ort is not None:
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            return
        self.net = cv2.dnn.readNetFromONNX(self.model_path)

    def _prepare_input(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 2:
            raise ValueError("SuperPointExtractor expects a grayscale image.")
        image = img.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        return image[None, None, :, :]

    def _model_input_size(self) -> Optional[tuple[int, int]]:
        if self.session is None:
            return None

        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) != 4:
            return None

        height = input_shape[2]
        width = input_shape[3]
        if isinstance(height, int) and isinstance(width, int):
            return height, width
        return None

    def _extract_score_map(
        self, raw_scores: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        scores = np.asarray(raw_scores, dtype=np.float32)

        if scores.ndim == 4 and scores.shape[0] == 1 and scores.shape[1] == 65:
            semi = scores[0]
            semi = semi - np.max(semi, axis=0, keepdims=True)
            prob = np.exp(semi)
            prob /= np.sum(prob, axis=0, keepdims=True) + 1e-12
            prob = prob[:-1, :, :]
            h8, w8 = prob.shape[1:]
            prob = prob.transpose(1, 2, 0).reshape(h8, w8, 8, 8)
            prob = prob.transpose(0, 2, 1, 3).reshape(h8 * 8, w8 * 8)
            return prob[:height, :width]

        if scores.ndim == 4 and scores.shape[0] == 1 and scores.shape[1] == 1:
            return scores[0, 0]

        if scores.ndim == 3 and scores.shape[0] == 1:
            return scores[0]

        raise ValueError(f"Unsupported SuperPoint score output shape: {scores.shape}")

    def _extract_dense_descriptors(self, raw_descriptors: np.ndarray) -> np.ndarray:
        descriptors = np.asarray(raw_descriptors, dtype=np.float32)

        if descriptors.ndim == 4 and descriptors.shape[0] == 1:
            return descriptors[0]

        if descriptors.ndim == 3:
            return descriptors

        raise ValueError(
            f"Unsupported SuperPoint descriptor output shape: {descriptors.shape}"
        )

    def _simple_nms(self, score_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(score_map >= self.keypoint_threshold)
        if len(xs) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        scores = score_map[ys, xs]
        order = np.argsort(scores)[::-1]
        ys = ys[order]
        xs = xs[order]

        kept_xs = []
        kept_ys = []
        suppressed = np.zeros(score_map.shape, dtype=bool)

        for x, y in zip(xs, ys):
            if suppressed[y, x]:
                continue
            kept_xs.append(x)
            kept_ys.append(y)

            y0 = max(0, y - self.nms_radius)
            y1 = min(score_map.shape[0], y + self.nms_radius + 1)
            x0 = max(0, x - self.nms_radius)
            x1 = min(score_map.shape[1], x + self.nms_radius + 1)
            suppressed[y0:y1, x0:x1] = True

            if len(kept_xs) >= self.max_keypoints:
                break

        return np.asarray(kept_xs, dtype=np.int32), np.asarray(kept_ys, dtype=np.int32)

    def _remove_border_keypoints(
        self, xs: np.ndarray, ys: np.ndarray, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(xs) == 0:
            return xs, ys

        border = self.remove_borders
        valid = (
            (xs >= border)
            & (xs < max(border, width - border))
            & (ys >= border)
            & (ys < max(border, height - border))
        )
        return xs[valid], ys[valid]

    def _sample_descriptors(
        self,
        dense_descriptors: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        if len(xs) == 0:
            return None

        desc_map = dense_descriptors.transpose(1, 2, 0)
        desc_map = cv2.resize(
            desc_map,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        descriptors = desc_map[ys, xs].astype(np.float32)
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / np.clip(norms, 1e-12, None)
        return descriptors

    def extract(self, img):
        self._load_net()
        original_height, original_width = img.shape[:2]
        resized_img = img
        model_input_size = self._model_input_size()
        if model_input_size is not None:
            model_height, model_width = model_input_size
            if (original_height, original_width) != (model_height, model_width):
                resized_img = cv2.resize(
                    img,
                    (model_width, model_height),
                    interpolation=cv2.INTER_AREA,
                )

        blob = self._prepare_input(resized_img)
        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: blob})
        else:
            self.net.setInput(blob)
            output_names = self.net.getUnconnectedOutLayersNames()
            outputs = (
                self.net.forward(output_names) if output_names else self.net.forward()
            )
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

        score_output = None
        descriptor_output = None
        for output in outputs:
            if not isinstance(output, np.ndarray):
                continue
            if output.ndim >= 3 and (
                (output.ndim == 4 and output.shape[1] in (1, 65))
                or (output.ndim == 3 and output.shape[0] == 1)
            ):
                score_output = output
            elif output.ndim >= 3:
                descriptor_output = output

        if score_output is None or descriptor_output is None:
            raise RuntimeError(
                "Could not parse SuperPoint outputs. Expected score map and dense descriptors."
            )

        height, width = resized_img.shape[:2]
        score_map = self._extract_score_map(score_output, height, width)
        dense_descriptors = self._extract_dense_descriptors(descriptor_output)

        xs, ys = self._simple_nms(score_map)
        xs, ys = self._remove_border_keypoints(xs, ys, width, height)
        if len(xs) == 0:
            return [], None

        scores = score_map[ys, xs]
        order = np.argsort(scores)[::-1]
        xs = xs[order]
        ys = ys[order]
        scores = scores[order]

        descriptors = self._sample_descriptors(
            dense_descriptors,
            xs,
            ys,
            width,
            height,
        )

        scale_x = original_width / width
        scale_y = original_height / height

        keypoints = [
            cv2.KeyPoint(float(x * scale_x), float(y * scale_y), 1, -1, float(score))
            for x, y, score in zip(xs, ys, scores)
        ]
        return keypoints, descriptors

    def __str__(self):
        return "SuperPoint"


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
        """Initialize panorama base.

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
        panorama2 = cv2.warpPerspective(
            query_img, H, (width_panorama, height_panorama)
        ).astype(np.uint8)
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
        kernel = np.ones((3, 3), np.uint8)
        mask2_eroded = cv2.erode(mask2, kernel, iterations=1)
        x, y, w, h = cv2.boundingRect(mask2_eroded)
        if w == 0 or h == 0:
            return self._post_processing(base_target)
        roi_src = panorama2[y : y + h, x : x + w]
        roi_dst = base_target[y : y + h, x : x + w]
        roi_mask = mask2_eroded[y : y + h, x : x + w]
        center = (w // 2, h // 2)
        try:
            blended_roi = cv2.seamlessClone(
                src=roi_src,
                dst=roi_dst,
                mask=roi_mask,
                p=center,
                flags=cv2.NORMAL_CLONE,
            )
        except Exception as e:
            print(f"[ERROR] Error at bleding process: {repr(e)}")
            blended_roi = roi_dst
        blended_img = base_target.copy()
        blended_img[y : y + h, x : x + w] = blended_roi
        return blended_roi, blended_img

    def __str__(self):
        return "Poisson Blending"


# ==============
# ==============
# ==============


class PanoramaConfig(TypedDict):
    extractor: FeatureExtractor
    blending: BlendingBase


class PanoramaStitchingPipeline(BasePipeline):
    """
    Panorama Image Stitching Pipeline.
    """

    def __init__(self, config: PanoramaConfig):
        self.extractor = config["extractor"]
        self.blending = config["blending"]

    @staticmethod
    def _metric_columns() -> List[str]:
        return [
            "num_matches",
            "num_inliers",
            "inlier_ratio",
            "precision",
            "recall",
            "mre",
            "rmse",
        ]

    @staticmethod
    def _build_image_labels(
        input_data: Union[str, List[str], List[np.ndarray], np.ndarray],
        count: int,
    ) -> List[str]:
        if isinstance(input_data, list) and all(
            isinstance(item, str) for item in input_data
        ):
            return [
                os.path.basename(path) or f"image_{index}"
                for index, path in enumerate(input_data[:count])
            ]
        return [f"image_{index}" for index in range(count)]

    @classmethod
    def _failure_metrics(cls, num_matches: int = 0) -> Dict[str, Any]:
        return {
            "num_matches": int(num_matches),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "precision": 0.0,
            "recall": np.nan,
            "mre": np.nan,
            "rmse": np.nan,
        }

    @classmethod
    def _metrics_to_row(
        cls,
        left_label: str,
        right_label: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "left_image": left_label,
            "right_image": right_label,
        }
        for column in cls._metric_columns():
            row[column] = metrics.get(column, np.nan)
        return row

    @staticmethod
    def _metric_frame(metric_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        if len(metric_rows) == 0:
            frame = pd.DataFrame(
                columns=[
                    "left_image",
                    "right_image",
                    *PanoramaStitchingPipeline._metric_columns(),
                ]
            )
            frame.index = pd.MultiIndex.from_tuples(
                [],
                names=["left_image", "right_image"],
            )
            return frame

        frame = pd.DataFrame(metric_rows)
        pair_index = list(zip(frame.pop("left_image"), frame.pop("right_image")))
        frame.index = pd.MultiIndex.from_tuples(
            pair_index,
            names=["left_image", "right_image"],
        )
        return frame

    @valid_input
    def run(
        self, input: Union[str, List[str], List[np.ndarray]], is_visual=True
    ) -> Dict[str, Any]:
        """
        Run PanoramaStitchingPipeline.

        Args:
            @input: either image path, list of image paths or image tensor.
        """
        images = self._read_input(input)
        image_labels = self._build_image_labels(input, len(images))
        start = time.time()
        try:
            if len(images) < 2:
                raise ValueError("At least 2 images are required for stitching")
            result = self._run_multiple_image(images, image_labels)
            end = time.time()
            result["time(s)"] = float(end - start)
            if is_visual:
                PanoramaStitchingPipeline.visualize(
                    result["result"], str(self.extractor)
                )
            return result

        except Exception as e:
            raise e

    def _preprocessing(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

    def _run_two_image(
        self,
        left_img,
        right_img,
        left_label: str,
        right_label: str,
    ):
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
            failure_metrics = self._failure_metrics(num_matches=0)
            return {
                "result": left_img,
                "metric_row": self._metrics_to_row(
                    left_label,
                    right_label,
                    failure_metrics,
                ),
            }

        matches = self._match_features(features_query, features_train)
        if len(matches) < 8:
            print(
                f"[ERROR] Not enough good matches after FLANN ration test: {len(matches)}"
            )
            failure_metrics = self._failure_metrics(num_matches=len(matches))
            return {
                "result": left_img,
                "metric_row": self._metrics_to_row(
                    left_label,
                    right_label,
                    failure_metrics,
                ),
            }

        # ======================================
        # Stage 4: Tìm ma trận Homo
        # ======================================
        try:
            _, H, status = PanoramaStitchingPipeline.compute_homography(
                keypoints_train, keypoints_query, matches, 5
            )
        except Exception:
            failure_metrics = self._failure_metrics(num_matches=len(matches))
            return {
                "result": left_img,
                "metric_row": self._metrics_to_row(
                    left_label,
                    right_label,
                    failure_metrics,
                ),
            }

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

        metrics = evaluate_from_matches(
            keypoints_train=keypoints_train,
            keypoints_query=keypoints_query,
            matches=matches,
            homography=H,
            status=status,
        )

        info = {
            "result": result_img,
            "metric_row": self._metrics_to_row(
                left_label,
                right_label,
                metrics,
            ),
        }
        return info

    def _run_multiple_image(self, images, image_labels: List[str]):
        if len(images) < 2:
            raise ValueError("At least 2 images are required for stitching")

        pairwise_results: List[Dict[str, Any]] = []
        metric_rows: List[Dict[str, Any]] = []

        current_image = images[0]
        current_label = image_labels[0] if len(image_labels) > 0 else "image_0"

        for index in range(1, len(images)):
            next_image = images[index]
            next_label = (
                image_labels[index] if index < len(image_labels) else f"image_{index}"
            )
            step_result = self._run_two_image(
                current_image,
                next_image,
                current_label,
                next_label,
            )
            pairwise_results.append(step_result)
            metric_rows.append(step_result["metric_row"])
            current_image = step_result["result"]
            current_label = f"panorama_{index}"

        return {
            "result": current_image,
            "pairwise_results": pairwise_results,
            "metric": self._metric_frame(metric_rows),
        }

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
    def visualize(image, name):
        plt.axis(False)
        plt.tight_layout()
        plt.imshow(image)
        plt.title("Image Stitching with " + name)
        plt.show()


if __name__ == "__main__":
    blending = PoissonBlending()
    # extractor = AKAZE()
    extractor = SuperPointExtractor()
    config = {"extractor": extractor, "blending": blending}
    pl = PanoramaStitchingPipeline(config)
    data_dir = os.path.abspath(r"img\btl3\BK")
    image_paths = [
        os.path.join(data_dir, file_name) for file_name in sorted(os.listdir(data_dir))
    ]
    result = pl.run(image_paths)
    from pprint import pprint

    pprint(result)
    # print(
    #     f'Num keypoints left: {result.get("kp_train", 0)} | Num keypoints right: {result.get("kp_query", 0)} | Num matches: {result.get("n_matches", 0)}'
    # )
    # # print(blending.accept(v))
