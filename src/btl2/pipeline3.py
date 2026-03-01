import cv2
import numpy as np
from matplotlib import pyplot as plt


class ProjectiveTransformPipeline:
	"""Pipeline for projective transformation to paste source image onto background"""
	def __init__(self):
		self.final_result = None
		self.warped_img = None
		self.mask = None
		self.bg_blacked_out = None
		self.warped_img_clean = None

	def run(self, pt: np.ndarray, bg: np.ndarray, pts_dst: np.array, inter_res = True):
		"""
		Apply projective transformation to paste source image onto background
		
		Args:
			pt: Source image (numpy array)
			bg: Background image (numpy array)
			dst_pts: Frame object containing 4 destination points on background
			
		Returns:
			final_result: Composited image and intermediate steps
		"""

		self.pt = pt
		self.bg = bg
		self.pts_dst = pts_dst
		h_src, w_src = pt.shape[:2]
		h_bg, w_bg = bg.shape[:2]
		
		# Step 1: Define source points (4 corners of source image)
		pts_src = np.float32([
			[0, 0],           # Top-left
			[w_src, 0],       # Top-right
			[w_src, h_src],   # Bottom-right
			[0, h_src]        # Bottom-left
		])
		
		# Step 2: Calculate homography matrix
		matrix_H = cv2.getPerspectiveTransform(pts_src, pts_dst)
		
		# Step 3: Warp source image
		self.warped_img = cv2.warpPerspective(pt, matrix_H, (w_bg, h_bg))
		
		# Step 4: Create mask for destination region
		self.mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
		cv2.fillConvexPoly(self.mask, pts_dst.astype(np.int32), 255)
  
		# Step 5: Create inverse mask
		mask_inv = cv2.bitwise_not(self.mask)
		
		# Step 6: "Cut hole" in background
		self.bg_blacked_out = cv2.bitwise_and(bg, bg, mask=mask_inv)
		
		# Step 7: Get clean warped image
		self.warped_img_clean = cv2.bitwise_and(self.warped_img, self.warped_img, mask=self.mask)
		
		# Step 8: Composite images
		self.final_result = cv2.add(self.bg_blacked_out, self.warped_img_clean)
		
		if inter_res == True:
			return self.final_result, self.warped_img, self.mask, self.bg_blacked_out, self.warped_img_clean
		else:
			return self.final_result
  
	def visualize_result(self, inter=True):
		"""
		Visualize results
		"""
		fig, axes = plt.subplots(2, 3, figsize=(15, 10))
		
		try:
			axes[0, 0].imshow(self.pt)
			axes[0, 0].set_title("Source Image")
			axes[0, 0].axis(False)
		except Exception as e:
			print(f"[ERROR] {e.__str__()}")
		
		if inter:
			try:
				bg_display = self.bg.copy()
				axes[0, 1].imshow(bg_display)
				for i, pt_coord in enumerate(self.pts_dst):
					axes[0, 1].plot(pt_coord[0], pt_coord[1], 'ro', markersize=8)
					axes[0, 1].text(pt_coord[0], pt_coord[1], f"  {i}", color='red', fontsize=10)
				axes[0, 1].set_title("Background with Destination Points")
				axes[0, 1].axis(False)
			except Exception as e:
				print(f"[ERROR] {e.__str__()}")
			try:
				axes[0, 2].imshow(self.warped_img)
				axes[0, 2].set_title("Warped Image (Intermediate)")
				axes[0, 2].axis(False)
			except Exception as e:
				print(f"[ERROR] {e.__str__()}")
			
			try:
				axes[1, 0].imshow(self.mask, cmap='gray')
				axes[1, 0].set_title("Mask")
				axes[1, 0].axis(False)
			except Exception as e:
				print(f"[ERROR] {e.__str__()}")
			
			try:
				axes[1, 1].imshow(self.bg_blacked_out)
				axes[1, 1].set_title("Background with Hole")
				axes[1, 1].axis(False)
			except Exception as e:
				print(f"[ERROR] {e.__str__()}")

			try:
				axes[1, 2].imshow(self.final_result)
				axes[1, 2].set_title("Final Result")
				axes[1, 2].axis(False)
			except Exception as e:
				print(f"[ERROR] {e.__str__()}")
		
		plt.tight_layout()
		plt.show()