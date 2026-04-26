import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import wraps
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional, Tuple

try:
    from .base_pipeline import *
    from .pipeline1 import *
    from .pipeline2 import *
    from .pipeline3 import *
    from .pipeline4 import *
except ImportError:
    from base_pipeline import *
    from pipeline1 import *
    from pipeline2 import *
    from pipeline3 import *
    from pipeline4 import *


class OverallSceneAnalysisPipeline(BasePipeline):
    """
    Scence Analysis Pipeline
    """
    def __init__(self):
        # Composition của các pipeline con
        self.preprocessor = DataPreprocessorPipeline(apply_smoothing=True)
        self.geometry_1 = GeometryFeaturePipeline(edge_detector=Canny(threshold1=50, threshold2=150),
                                       corner_detector=ShiTomasiAlgo(),
                                       line_detector=LineDetector(150, 100, 10))
        self.geometry_2 = PanoramaStitchingPipeline({"extractor": SIFT(), "blending": AlphaBlending()})
        self.detection = ObjectDetectionPipeline()
        self.segmentation = SegmentationPipeline()

        self.last_results = {}

    def run(
        self,
        input: Union[str, List[str], List[np.ndarray]],
        visualize: bool = True,
        run_geometry_1: bool = True,
        run_geometry_2: bool = True,
        run_segmentation: bool = True,
        run_detection: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all pipeline and return each pipeline's result.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        res = {}
        res['preprocess'] = self.preprocessor.run(input, visualize=False)

        preprocessed_rgb = res['preprocess']["preprocessed_imgs"]

        if run_geometry_1:
            res['geometry_1'] = self.geometry_1.run(preprocessed_rgb)
        
        if run_geometry_2:
            res["stitching"] = self.geometry_2.run(preprocessed_rgb)
        stitching_img = [res["stitching"]["result"]]
        if run_detection:
            res['detection'] = self.detection.run(stitching_img, visualize=visualize)
            
        if run_segmentation:
            res['segmentation'] = self.segmentation.run(stitching_img, visualize=visualize)

        self.last_results = res
        return res

if __name__ == "__main__":
    data_dir = os.path.abspath(r"img\btl4\Overall")
    pipeline = OverallSceneAnalysisPipeline()
    pipeline.run([os.path.join(data_dir, dir) for dir in os.listdir(data_dir)])