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
        self.preprocessor = DataPreprocessorPipeline()
        self.geometry_1 = GeometryFeaturePipeline()
        self.geometry_2 = PanoramaStitchingPipeline()
        self.detection = ObjectDetectionPipeline()
        self.segmentation = SegmentationPipeline()

        self.last_results = {}

    def run(self, input: Union[str, List[str], List[np.ndarray]], visualize=True) -> Dict[str, Dict[str, Any]]:
        """
        Run all pipeline and return each pipeline's result.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        res = {}
        res['preprocess'] = self.preprocessor.run(input)

        preprocessed_rgb = res['preprocess']["rgb_images"]
        self.visualize(res['preprocess'], "rgb_images", "Smooth Image")

        res['geometry_1'] = self.geometry_1.run(preprocessed_rgb)
        # res['geometry_2'] = self.geometry_2.run(preprocessed_rgb)
        # res['detection'] = self.detection.run(preprocessed_rgb)
        # res['segmentation'] = self.segmentation.run(preprocessed_rgb)       

if __name__ == "__main__":
    data_dir = os.path.abspath(r"img\btl4\GeometryFeature")
    pipeline = OverallSceneAnalysisPipeline()
    pipeline.run([os.path.join(data_dir, dir) for dir in os.listdir(data_dir)])