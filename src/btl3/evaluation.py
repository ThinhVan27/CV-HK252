from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - optional dependency
    skimage_psnr = None
    skimage_ssim = None


def _to_points(points: np.ndarray | list[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2).")
    return arr


def _to_float_image(image: np.ndarray) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError("Image must be 2D (grayscale) or 3D (color).")
    return image.astype(np.float64)


def _ensure_same_shape(image_a: np.ndarray, image_b: np.ndarray) -> None:
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Input images must have the same shape. Got {image_a.shape} and {image_b.shape}."
        )


def project_points(points_src: np.ndarray | list[tuple[float, float]], homography: np.ndarray) -> np.ndarray:
    """Project 2D points using a homography matrix H."""
    src = _to_points(points_src).astype(np.float32)
    if homography.shape != (3, 3):
        raise ValueError("Homography matrix must have shape (3, 3).")
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography.astype(np.float64))
    return projected.reshape(-1, 2).astype(np.float64)


def reprojection_errors(
    points_src: np.ndarray | list[tuple[float, float]],
    points_dst: np.ndarray | list[tuple[float, float]],
    homography: np.ndarray,
) -> np.ndarray:
    """Compute per-point reprojection error d_i."""
    src = _to_points(points_src)
    dst = _to_points(points_dst)
    if src.shape[0] != dst.shape[0]:
        raise ValueError("points_src and points_dst must contain the same number of points.")
    projected = project_points(src, homography)
    return np.linalg.norm(projected - dst, axis=1)


def mean_reprojection_error(errors: np.ndarray) -> float:
    if errors.size == 0:
        return float("nan")
    return float(np.mean(errors))


def rmse_from_errors(errors: np.ndarray) -> float:
    if errors.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(errors))))


def alignment_metrics(
    points_src: np.ndarray | list[tuple[float, float]],
    points_dst: np.ndarray | list[tuple[float, float]],
    homography: np.ndarray,
) -> dict[str, Any]:
    """Return alignment metrics: per-point errors, MRE, RMSE."""
    errors = reprojection_errors(points_src, points_dst, homography)
    return {
        "reprojection_errors": errors,
        "mre": mean_reprojection_error(errors),
        "rmse": rmse_from_errors(errors),
    }


def nonzero_mask(image: np.ndarray, empty_value: int | float = 0) -> np.ndarray:
    if image.ndim == 2:
        return image != empty_value
    if image.ndim == 3:
        return np.any(image != empty_value, axis=2)
    raise ValueError("Image must be 2D or 3D.")


def overlap_mask(image_a: np.ndarray, image_b: np.ndarray, empty_value: int | float = 0) -> np.ndarray:
    _ensure_same_shape(image_a, image_b)
    return nonzero_mask(image_a, empty_value) & nonzero_mask(image_b, empty_value)


def mse(image_true: np.ndarray, image_test: np.ndarray, mask: np.ndarray | None = None) -> float:
    _ensure_same_shape(image_true, image_test)
    true = _to_float_image(image_true)
    test = _to_float_image(image_test)
    if mask is None:
        return float(np.mean((true - test) ** 2))

    if mask.shape != true.shape[:2]:
        raise ValueError(
            f"Mask shape must be {true.shape[:2]}. Got {mask.shape}."
        )

    valid = mask.astype(bool)
    if not np.any(valid):
        return float("nan")

    if true.ndim == 2:
        diff = true[valid] - test[valid]
    else:
        diff = true[valid, :] - test[valid, :]
    return float(np.mean(diff ** 2))


def psnr(
    image_true: np.ndarray,
    image_test: np.ndarray,
    data_range: float = 255.0,
    mask: np.ndarray | None = None,
) -> float:
    if mask is None and skimage_psnr is not None:
        return float(skimage_psnr(image_true, image_test, data_range=data_range))

    mse_value = mse(image_true, image_test, mask=mask)
    if mse_value == 0:
        return float("inf")
    if np.isnan(mse_value):
        return float("nan")
    return float(10.0 * np.log10((data_range ** 2) / mse_value))


def _ssim_single_channel(x: np.ndarray, y: np.ndarray, data_range: float = 255.0) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    if denominator == 0:
        return float("nan")
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    return float(numerator / denominator)


def ssim(
    image_true: np.ndarray,
    image_test: np.ndarray,
    data_range: float = 255.0,
    mask: np.ndarray | None = None,
) -> float:
    _ensure_same_shape(image_true, image_test)
    true = _to_float_image(image_true)
    test = _to_float_image(image_test)

    if mask is None and skimage_ssim is not None:
        channel_axis = -1 if true.ndim == 3 else None
        return float(skimage_ssim(true, test, data_range=data_range, channel_axis=channel_axis))

    if mask is not None and mask.shape != true.shape[:2]:
        raise ValueError(f"Mask shape must be {true.shape[:2]}. Got {mask.shape}.")

    valid = None if mask is None else mask.astype(bool)

    if true.ndim == 2:
        x = true if valid is None else true[valid]
        y = test if valid is None else test[valid]
        return _ssim_single_channel(x, y, data_range=data_range)

    scores: list[float] = []
    for channel in range(true.shape[2]):
        x = true[:, :, channel] if valid is None else true[:, :, channel][valid]
        y = test[:, :, channel] if valid is None else test[:, :, channel][valid]
        scores.append(_ssim_single_channel(x, y, data_range=data_range))
    return float(np.mean(scores))


def visual_quality_metrics(
    image_true: np.ndarray,
    image_test: np.ndarray,
    data_range: float = 255.0,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    return {
        "mse": mse(image_true, image_test, mask=mask),
        "psnr": psnr(image_true, image_test, data_range=data_range, mask=mask),
        "ssim": ssim(image_true, image_test, data_range=data_range, mask=mask),
    }


def inlier_ratio(num_inliers: int, num_matches: int) -> float:
    if num_matches <= 0:
        return 0.0
    return float((num_inliers / num_matches) * 100.0)


def precision(tp: int, fp: int) -> float:
    denominator = tp + fp
    if denominator <= 0:
        return 0.0
    return float(tp / denominator)


def recall(tp: int, fn: int) -> float:
    denominator = tp + fn
    if denominator <= 0:
        return 0.0
    return float(tp / denominator)


def feature_performance_metrics(
    num_inliers: int,
    num_matches: int,
    num_ground_truth_positives: int | None = None,
) -> dict[str, float]:
    tp = int(num_inliers)
    fp = int(max(num_matches - num_inliers, 0))
    result: dict[str, float] = {
        "inlier_ratio": inlier_ratio(num_inliers, num_matches),
        "precision": precision(tp, fp),
    }

    if num_ground_truth_positives is not None:
        fn = int(max(num_ground_truth_positives - tp, 0))
        result["recall"] = recall(tp, fn)
    else:
        result["recall"] = float("nan")

    return result


def alignment_from_matches(
    keypoints_train: list[cv2.KeyPoint],
    keypoints_query: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    homography: np.ndarray,
    status: np.ndarray | None = None,
) -> dict[str, Any]:
    if len(matches) == 0:
        raise ValueError("matches is empty.")

    points_train = np.float32([keypoints_train[m.trainIdx].pt for m in matches])
    points_query = np.float32([keypoints_query[m.queryIdx].pt for m in matches])

    if status is not None:
        inlier_mask = status.ravel().astype(bool)
        points_train = points_train[inlier_mask]
        points_query = points_query[inlier_mask]

    return alignment_metrics(points_train, points_query, homography)


def evaluate_from_matches(
    keypoints_train: list[cv2.KeyPoint],
    keypoints_query: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    homography: np.ndarray,
    status: np.ndarray | None = None,
    num_ground_truth_positives: int | None = None,
) -> dict[str, Any]:
    alignment = alignment_from_matches(
        keypoints_train=keypoints_train,
        keypoints_query=keypoints_query,
        matches=matches,
        homography=homography,
        status=status,
    )

    num_matches = len(matches)
    num_inliers = int(status.ravel().sum()) if status is not None else num_matches
    feature = feature_performance_metrics(
        num_inliers=num_inliers,
        num_matches=num_matches,
        num_ground_truth_positives=num_ground_truth_positives,
    )

    return {
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        **feature,
        **alignment,
    }
