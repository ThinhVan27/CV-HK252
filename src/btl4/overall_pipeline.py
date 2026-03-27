import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import wraps
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional, Tuple

from .base_pipeline import *
from .pipeline1 import *
from .pipeline2 import *
from .pipeline3 import *
from .pipeline4 import *


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
        self.viz = VisualizationPipeline()

        self.last_results = {}

    def run(self, input: Union[str, List[str], List[np.ndarray]], visualize=True) -> Dict[str, Dict[str, Any]]:
        """
        Run all pipeline and return each pipeline's result.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        res = {}
        # TODO
        # res['preprocess'] = self.preprocessor.run(input)
        # res['geometry_1'] = self.geometry_1.run(input)
        # res['geometry_2'] = self.geometry_2.run(input)
        # res['detection'] = self.detection.run(input)
        # res['segmentation'] = self.segmentation.run(input)
        
        self.last_results = res
        if visualize:
            self.visualize_all()
        return self.last_results

    def visualize_all(self):
        if self.last_results:
            self.viz.run(self.last_results)
        else:
            print("Have no results to visualize.")       