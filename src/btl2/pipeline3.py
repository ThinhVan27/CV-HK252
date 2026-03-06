import cv2
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Union, Optional

@dataclass
class Frame:
    tl: Tuple[int, int]
    tr: Tuple[int, int]
    br: Tuple[int, int]
    bl: Tuple[int, int]
    
    def to_array(self):
        return np.array([list(self.tl), list(self.tr), list(self.br), list(self.bl)], dtype=np.float32)


class ProjectiveTransformPipeline:
    """Stateless pipeline for projective transformation to paste source image onto background"""
    
    def run(self, src_img_path: Union[str, np.ndarray], bg_img_path: Union[str, np.ndarray], 
            frame: Frame, visualize: bool = False) -> np.ndarray:
        """
        Apply projective transformation to paste source image onto background
        
        Args:
            src_img_path: Path to source image or numpy array
            bg_img_path: Path to background image or numpy array
            frame: Frame object containing 4 destination points on background
            visualize: whether to display the results. Default: False
            
        Returns:
            final_result: Composited image
        """
        # Step 1: Prepare images
        src_img = self._prepare_image(src_img_path)
        bg_img = self._prepare_image(bg_img_path)
        pts_dst = frame.to_array()
        
        # Step 2: Calculate homography
        matrix_H = self._calc_homography(src_img, pts_dst)
        
        # Step 3: Warp and compose
        warped_img_clean, bg_blacked_out, final_result = self._compose(
            src_img, bg_img, matrix_H, pts_dst
        )
        
        # Step 4: Visualize if requested
        if visualize:
            self._visualize_result(src_img, bg_img, pts_dst, warped_img_clean, 
                                   bg_blacked_out, final_result)
        
        return final_result
    
    def _prepare_image(self, img_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Prepare image from path or array
        
        Args:
            img_input: Image path (str) or numpy array
            
        Returns:
            Image as numpy array
        """
        if isinstance(img_input, str):
            img = cv2.imread(img_input)
            if img is None:
                raise ValueError(f"Cannot read image from {img_input}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, np.ndarray):
            return img_input
        else:
            raise TypeError("img_input must be a file path (str) or numpy array")
    
    def _calc_homography(self, src_img: np.ndarray, pts_dst: np.ndarray) -> np.ndarray:
        """
        Calculate homography matrix
        
        Args:
            src_img: Source image
            pts_dst: Destination points (4 corners)
            
        Returns:
            Homography matrix
        """
        h_src, w_src = src_img.shape[:2]
        
        # Define source points (4 corners of source image)
        pts_src = np.float32([
            [0, 0],           # Top-left
            [w_src, 0],       # Top-right
            [w_src, h_src],   # Bottom-right
            [0, h_src]        # Bottom-left
        ])
        
        matrix_H = cv2.getPerspectiveTransform(pts_src, pts_dst)
        return matrix_H
    
    def _compose(self, src_img: np.ndarray, bg_img: np.ndarray, matrix_H: np.ndarray, 
                pts_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Warp source image and composite onto background
        
        Args:
            src_img: Source image
            bg_img: Background image
            matrix_H: Homography matrix
            pts_dst: Destination points
            
        Returns:
            Tuple of (warped_image_clean, bg_blacked_out, final_result)
        """
        h_bg, w_bg = bg_img.shape[:2]
        
        # Warp source image
        warped_img = cv2.warpPerspective(src_img, matrix_H, (w_bg, h_bg))
        
        # Create mask for destination region
        mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts_dst.astype(np.int32), 255)
        
        # Create inverse mask
        mask_inv = cv2.bitwise_not(mask)
        
        # Cut hole in background
        bg_blacked_out = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
        
        # Get clean warped image
        warped_img_clean = cv2.bitwise_and(warped_img, warped_img, mask=mask)
        
        # Composite images
        final_result = cv2.add(bg_blacked_out, warped_img_clean)
        
        return warped_img_clean, bg_blacked_out, final_result
    
    def _visualize_result(self, src_img: np.ndarray, bg_img: np.ndarray, pts_dst: np.ndarray,
                          warped_img_clean: np.ndarray, bg_blacked_out: np.ndarray, 
                          final_result: np.ndarray) -> None:
        """
        Visualize results
        
        Args:
            src_img: Source image
            bg_img: Background image
            pts_dst: Destination points
            warped_img_clean: Clean warped image
            bg_blacked_out: Background with hole
            final_result: Final composited result
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        try:
            axes[0, 0].imshow(src_img)
            axes[0, 0].set_title("Source Image")
            axes[0, 0].axis(False)
        except Exception as e:
            print(f"[ERROR] {e.__str__()}")
        
        try:
            bg_display = bg_img.copy()
            axes[0, 1].imshow(bg_display)
            for i, pt_coord in enumerate(pts_dst):
                axes[0, 1].plot(pt_coord[0], pt_coord[1], 'ro', markersize=8)
                axes[0, 1].text(pt_coord[0], pt_coord[1], f"  {i}", color='red', fontsize=10)
            axes[0, 1].set_title("Background with Destination Points")
            axes[0, 1].axis(False)
        except Exception as e:
            print(f"[ERROR] {e.__str__()}")
        
        try:
            axes[0, 2].imshow(warped_img_clean)
            axes[0, 2].set_title("Warped Image (Clean)")
            axes[0, 2].axis(False)
        except Exception as e:
            print(f"[ERROR] {e.__str__()}")
        
        try:
            mask = np.zeros_like(bg_img[:, :, 0])
            cv2.fillConvexPoly(mask, pts_dst.astype(np.int32), 255)
            axes[1, 0].imshow(mask, cmap='gray')
            axes[1, 0].set_title("Mask")
            axes[1, 0].axis(False)
        except Exception as e:
            print(f"[ERROR] {e.__str__()}")
        
        try:
            axes[1, 1].imshow(bg_blacked_out)
            axes[1, 1].set_title("Background with Hole")
            axes[1, 1].axis(False)
        except Exception as e:
            print(f"[ERROR] {e.__str__()}")

        try:
            axes[1, 2].imshow(final_result)
            axes[1, 2].set_title("Final Result")
            axes[1, 2].axis(False)
        except Exception as e:
            print(f"[ERROR] {e.__str__()}")

        plt.tight_layout()
        plt.show()