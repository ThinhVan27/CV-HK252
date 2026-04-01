from __future__ import annotations

from abc import ABC
from typing import Any

import cv2
import numpy as np
from matplotlib import pyplot as plt

try:
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - optional dependency
    skimage_psnr = None
    skimage_ssim = None


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _match_features_flann(
    descriptors_query: np.ndarray,
    descriptors_train: np.ndarray,
    ratio_thresh: float = 0.75,
) -> list[cv2.DMatch]:
    if descriptors_query is None or descriptors_train is None:
        return []
    if len(descriptors_query) < 2 or len(descriptors_train) < 2:
        return []

    knn_matches: list[list[cv2.DMatch]]
    if descriptors_query.dtype == np.uint8 and descriptors_train.dtype == np.uint8:
        try:
            index_params = {
                "algorithm": 6,
                "table_number": 6,
                "key_size": 12,
                "multi_probe_level": 1,
            }
            search_params = {"checks": 50}
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            knn_matches = matcher.knnMatch(descriptors_query, descriptors_train, k=2)
        except cv2.error:
            bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            knn_matches = bf_matcher.knnMatch(descriptors_query, descriptors_train, k=2)
    else:
        query_float = np.asarray(descriptors_query, dtype=np.float32)
        train_float = np.asarray(descriptors_train, dtype=np.float32)
        try:
            index_params = {"algorithm": 1, "trees": 5}
            search_params = {"checks": 50}
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            knn_matches = matcher.knnMatch(query_float, train_float, k=2)
        except cv2.error:
            bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            knn_matches = bf_matcher.knnMatch(query_float, train_float, k=2)

    good_matches: list[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return sorted(good_matches, key=lambda x: x.distance)


def _draw_matches_with_thickness(
    left_image: np.ndarray,
    keypoints_left: list[cv2.KeyPoint],
    right_image: np.ndarray,
    keypoints_right: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    line_thickness: int = 2,
) -> np.ndarray:
    thickness = max(1, int(line_thickness))
    try:
        return cv2.drawMatches(
            left_image,
            keypoints_left,
            right_image,
            keypoints_right,
            matches,
            None,
            matchesThickness=thickness,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    except TypeError:
        vis = cv2.drawMatches(
            left_image,
            keypoints_left,
            right_image,
            keypoints_right,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        if thickness <= 1:
            return vis

        x_offset = left_image.shape[1]
        for match in matches:
            p1 = np.round(keypoints_left[match.queryIdx].pt).astype(int)
            p2 = np.round(keypoints_right[match.trainIdx].pt).astype(int)
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0] + x_offset), int(p2[1]))
            cv2.line(vis, pt1, pt2, (0, 255, 255), thickness, lineType=cv2.LINE_AA)
        return vis


def _swap_match_indices(matches: list[cv2.DMatch]) -> list[cv2.DMatch]:
    """Return matches with query/train indices swapped for display alignment."""
    swapped: list[cv2.DMatch] = []
    for match in matches:
        try:
            swapped_match = cv2.DMatch(
                _queryIdx=int(match.trainIdx),
                _trainIdx=int(match.queryIdx),
                _imgIdx=int(match.imgIdx),
                _distance=float(match.distance),
            )
        except TypeError:
            swapped_match = cv2.DMatch(
                int(match.trainIdx),
                int(match.queryIdx),
                int(match.imgIdx),
                float(match.distance),
            )
        swapped.append(swapped_match)
    return swapped


def visualize_extracted_features(
    image: np.ndarray,
    extractor: Any,
    extractor_name: str,
    max_keypoints: int = 300,
    figsize: tuple[float, float] = (8, 6),
    show: bool = True,
) -> dict[str, Any]:
    """Draw keypoints extracted by a single feature extractor."""
    gray = _gray(image)
    keypoints, descriptors = extractor.extract(gray)
    keypoints = keypoints or []
    selected = keypoints[:max_keypoints]
    num_keypoints_total = int(len(keypoints))
    num_keypoints_drawn = int(len(selected))

    vis = cv2.drawKeypoints(
        image,
        selected,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    if show:
        plt.figure(figsize=figsize)
        plt.imshow(_to_rgb(vis))
        plt.axis("off")
        plt.title(f"{extractor_name}: shown {num_keypoints_drawn}/{num_keypoints_total} keypoints")
        plt.tight_layout()
        plt.show()

    return {
        "extractor": extractor_name,
        "num_keypoints": num_keypoints_total,
        "num_keypoints_drawn": num_keypoints_drawn,
        "descriptor_shape": None if descriptors is None else tuple(descriptors.shape),
        "keypoints": keypoints,
        "descriptors": descriptors,
        "visualization": vis,
    }


def compare_extractors_features(
    image: np.ndarray,
    extractors: dict[str, Any],
    max_keypoints: int = 250,
    cols: int = 2,
    figsize_per_plot: tuple[float, float] = (6, 4),
    show: bool = True,
) -> dict[str, dict[str, Any]]:
    """Visualize keypoints from multiple extractors in a grid."""
    results: dict[str, dict[str, Any]] = {}
    for name, extractor in extractors.items():
        results[name] = visualize_extracted_features(
            image=image,
            extractor=extractor,
            extractor_name=name,
            max_keypoints=max_keypoints,
            show=False,
        )

    if show and len(results) > 0:
        names = list(results.keys())
        rows = int(np.ceil(len(names) / max(cols, 1)))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows),
        )
        axes = np.array(axes, dtype=object).reshape(-1)
        for i, name in enumerate(names):
            axes[i].imshow(_to_rgb(results[name]["visualization"]))
            axes[i].axis("off")
            axes[i].set_title(
                f"{name} ({results[name]['num_keypoints_drawn']}/{results[name]['num_keypoints']} kp)"
            )
        for i in range(len(names), len(axes)):
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()

    return results


def visualize_feature_matches(
    left_image: np.ndarray,
    right_image: np.ndarray,
    extractor: Any,
    extractor_name: str,
    max_matches_to_draw: int = 80,
    ratio_thresh: float = 0.75,
    show: bool = True,
    figsize: tuple[float, float] = (14, 6),
    match_line_thickness: int = 2,
) -> dict[str, Any]:
    """Draw line-based matches between two images for a feature extractor."""
    left_gray = _gray(left_image)
    right_gray = _gray(right_image)

    kp_train, desc_train = extractor.extract(left_gray)
    kp_query, desc_query = extractor.extract(right_gray)
    kp_train = kp_train or []
    kp_query = kp_query or []

    if desc_train is None or desc_query is None:
        raise RuntimeError(f"{extractor_name} failed to compute descriptors for matching.")

    raw_matches = _match_features_flann(desc_query, desc_train, ratio_thresh=ratio_thresh)
    matches = _swap_match_indices(raw_matches)
    max_draw = max(0, int(max_matches_to_draw))
    num_matches_drawn = int(min(len(raw_matches), max_draw))
    draw_matches = matches[:max_draw]
    matches_vis = _draw_matches_with_thickness(
        left_image,
        kp_train,
        right_image,
        kp_query,
        draw_matches,
        line_thickness=match_line_thickness,
    )

    if show:
        plt.figure(figsize=figsize)
        plt.imshow(_to_rgb(matches_vis))
        plt.axis("off")
        plt.title(f"{extractor_name}: shown {num_matches_drawn}/{len(raw_matches)} good matches")
        plt.tight_layout()
        plt.show()

    return {
        "extractor": extractor_name,
        "num_keypoints_left": int(len(kp_train)),
        "num_keypoints_right": int(len(kp_query)),
        "num_matches": int(len(raw_matches)),
        "num_matches_drawn": num_matches_drawn,
        "matches": matches,
        "keypoints_left": kp_train,
        "keypoints_right": kp_query,
        "visualization": matches_vis,
    }


def evaluate_stitching_pair(
    left_image: np.ndarray,
    right_image: np.ndarray,
    extractor: Any,
    blending: Any,
    extractor_name: str,
    blending_name: str,
    ratio_thresh: float = 0.75,
    reproj_thresh: float = 2.0,
    max_matches_to_draw: int = 80,
    match_line_thickness: int = 2,
    show: bool = True,
    figsize: tuple[float, float] = (18, 5),
) -> dict[str, Any]:
    """Run full two-image stitching and return quantitative + visual metrics."""
    left_gray = _gray(left_image)
    right_gray = _gray(right_image)

    keypoints_train, features_train = extractor.extract(left_gray)
    keypoints_query, features_query = extractor.extract(right_gray)
    keypoints_train = keypoints_train or []
    keypoints_query = keypoints_query or []

    if features_train is None or features_query is None:
        raise RuntimeError("Descriptor extraction failed, cannot evaluate stitching.")

    matches = _match_features_flann(features_query, features_train, ratio_thresh=ratio_thresh)
    if len(matches) < 4:
        raise RuntimeError(f"Not enough matches for homography: {len(matches)}")

    points_train = np.float32([keypoints_train[m.trainIdx].pt for m in matches])
    points_query = np.float32([keypoints_query[m.queryIdx].pt for m in matches])
    homography, status = cv2.findHomography(points_train, points_query, cv2.RANSAC, reproj_thresh)
    if homography is None:
        raise RuntimeError("Homography estimation failed.")

    homography_inv = np.linalg.inv(homography)
    stitched = blending.blend(left_image, right_image, homography_inv)

    blending_metrics: dict[str, float | int] = {}
    if hasattr(blending, "accept"):
        try:
            blending_metrics = BlendingVisitor().visit(blending)
        except Exception:
            blending_metrics = {}

    metrics = evaluate_from_matches(
        keypoints_train=keypoints_train,
        keypoints_query=keypoints_query,
        matches=matches,
        homography=homography,
        status=status,
    )
    metrics.update(blending_metrics)
    metrics["extractor"] = extractor_name
    metrics["blending"] = blending_name

    # FLANN is called as (query=right, train=left), so swap indices for left->right drawing.
    draw_matches = _swap_match_indices(matches[: max(0, int(max_matches_to_draw))])
    match_vis = _draw_matches_with_thickness(
        left_image,
        keypoints_train,
        right_image,
        keypoints_query,
        draw_matches,
        line_thickness=match_line_thickness,
    )

    if show:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(_to_rgb(left_image))
        axes[0].axis("off")
        axes[0].set_title("Left image")
        axes[1].imshow(_to_rgb(match_vis))
        axes[1].axis("off")
        axes[1].set_title(f"Matches ({len(matches)})")
        axes[2].imshow(_to_rgb(stitched))
        axes[2].axis("off")
        axes[2].set_title(f"Stitched ({extractor_name} + {blending_name})")
        plt.tight_layout()
        plt.show()

    return {
        "metrics": metrics,
        "status": status,
        "homography": homography,
        "stitched_image": stitched,
        "matches_visualization": match_vis,
        "matches": matches,
        "keypoints_train": keypoints_train,
        "keypoints_query": keypoints_query,
    }


def benchmark_extractors_and_blending(
    left_image: np.ndarray,
    right_image: np.ndarray,
    extractors: dict[str, Any],
    blending_factories: dict[str, Any],
    ratio_thresh: float = 0.75,
    reproj_thresh: float = 2.0,
    show: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate all extractor/blending combinations for one image pair."""
    records: list[dict[str, Any]] = []
    for extractor_name, extractor in extractors.items():
        for blending_name, blending_factory in blending_factories.items():
            blending = blending_factory()
            try:
                result = evaluate_stitching_pair(
                    left_image=left_image,
                    right_image=right_image,
                    extractor=extractor,
                    blending=blending,
                    extractor_name=extractor_name,
                    blending_name=blending_name,
                    ratio_thresh=ratio_thresh,
                    reproj_thresh=reproj_thresh,
                    show=show,
                )
                row = dict(result["metrics"])
                row["stitched_image"] = result["stitched_image"]
                records.append(row)
            except Exception as exc:
                records.append(
                    {
                        "extractor": extractor_name,
                        "blending": blending_name,
                        "error": str(exc),
                    }
                )
    return records


def visualize_stitching_gallery(
    benchmark_records: list[dict[str, Any]],
    cols: int = 2,
    figsize_per_plot: tuple[float, float] = (8, 5),
) -> None:
    """Visualize stitched outputs from benchmark records."""
    valid = [row for row in benchmark_records if "stitched_image" in row]
    if len(valid) == 0:
        return

    rows = int(np.ceil(len(valid) / max(1, cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows))
    axes = np.array(axes, dtype=object).reshape(-1)
    for i, row in enumerate(valid):
        axes[i].imshow(_to_rgb(row["stitched_image"]))
        axes[i].axis("off")
        axes[i].set_title(f"{row['extractor']} + {row['blending']}")
    for i in range(len(valid), len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


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


class BaseVisitor(ABC):
    def visit(self, obj: Any):
        if not hasattr(obj, "accept"):
            raise TypeError("Visited object must implement accept(visitor).")
        return obj.accept(self)

class BlendingVisitor(BaseVisitor):
    """Evaluate SSIM on overlap region between two panoramas after homography warp."""

    def __init__(self, data_range: float = 255.0, empty_value: int | float = 0):
        self.data_range = float(data_range)
        self.empty_value = empty_value

    def _evaluate_overlap_ssim(
        self,
        panorama1: np.ndarray,
        panorama2: np.ndarray,
    ) -> dict[str, float | int]:
        _ensure_same_shape(panorama1, panorama2)

        overlap = overlap_mask(panorama1, panorama2, empty_value=self.empty_value)
        overlap_pixels = int(np.count_nonzero(overlap))
        overlap_ratio = float(overlap_pixels / overlap.size) if overlap.size > 0 else float("nan")

        if overlap_pixels == 0:
            score = float("nan")
        else:
            score = ssim(
                image_true=panorama1,
                image_test=panorama2,
                data_range=self.data_range,
                mask=overlap,
            )

        return {
            "ssim_overlap": float(score),
            "overlap_pixels": overlap_pixels,
            "overlap_ratio": overlap_ratio,
        }

    def visitAlpha(self, blending):
        if not hasattr(blending, "panorama"):
            raise AttributeError(
                "Alpha blending object has no panorama cache. "
                "Please store (panorama1, panorama2) to blending.panorama before visiting."
            )
        panorama1, panorama2 = blending.panorama
        return self._evaluate_overlap_ssim(panorama1, panorama2)

    def visitPoisson(self, blending):
        if not hasattr(blending, "panorama"):
            raise AttributeError(
                "Poisson blending object has no panorama cache. "
                "Run blend() first so blending.panorama is available."
            )
        panorama1, panorama2 = blending.panorama
        return self._evaluate_overlap_ssim(panorama1, panorama2)
        