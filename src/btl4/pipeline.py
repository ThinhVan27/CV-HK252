import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional

def valid_input(func):
    def wrapper(*args):
        input = args[0]
        def valid_path(path):
            if not os.path.exists(path):
                raise ValueError(f"{path} do not exists")
            if not path.endswith((".jpg", ".png", ".jpeg")):
                raise ValueError(f"{path} is not an image file")
        if isinstance(input, str):
            valid_path(input)
        elif isinstance(input, list[str]):
            for path in input:
                valid_path(path)
        elif isinstance(input, np.ndarray):
            if len(np.shape(input)) not in [3, 4]:
                raise ValueError(f"Invalid image array")
        return func(*args)
    return wrapper
            
            
class BasePipeline(ABC):
    """
    Base Pipeline.
    """
    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Run pipeline.
        
        Returns:
          @result
        """
        pass


class DataPreprocessorPipeline(BasePipeline):
    """
    Image Preprocessing Pipeline
    """
    def __init__(self):
        pass

    @valid_input
    def run(self, input: Union[str, List[str], np.ndarray]) -> Dict[str, Any]:
        """
        Run DataPreprocessorPipeline.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """


class EdgeLineCornerPipeline(BasePipeline):
    """
    Boundaries, corners and edges detection pipeline.
    """
    
    @valid_input
    def run(self, input: Union[str, List[str], np.ndarray]) -> Dict[str, Any]:
        """
        Run EdgeLineCornerPipeline.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        pass


class PanoramaStitchingPipeline(BasePipeline):
    """
    Panorama Image Stitching Pipeline.
    """
    
    @valid_input
    def run(self, input: Union[str, List[str], np.ndarray]) -> Dict[str, Any]:
        """
        Run PanoramaStitchingPipeline.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        pass


class ObjectDetectionPipeline(BasePipeline):
    """
    Object Detection Pipeline
    """
    def __init__(self):
        pass

    @valid_input
    def run(self, input: Union[str, List[str], np.ndarray]) -> Dict[str, Any]:
        """
        Run `OjectDetectionPipeline`.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        pass
    
    
class SegmentationPipeline(BasePipeline):
    """
    Segmentation Pipeline
    """
    
    @valid_input
    def run(self, input: Union[str, List[str], np.ndarray]) -> Dict[str, Any]:
        """
        Run `SegmentationPipeline`.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        pass


class VisualizationPipeline(BasePipeline):
    """
    Visualization Pipeline.
    """
    def run(self, results: Dict[str, Dict[str, Any]]):
        for name, result in results.items():
            n = len(result)
            fig, axes = plt.subplots(n//3+1, 3, figsize=(18, 10))
            axes = axes.flatten()
            for i, ax in enumerate(axes):
                ax.imshow(result[i])
                ax.set_title(result.keys()[i])
            plt.title(f"{name} Results Visualization")
        plt.tight_layout()
        plt.show()


class OverallSceneAnalysisPipeline(BasePipeline):
    """
    Scence Analysis Pipeline
    """
    def __init__(self):
        # Composition của các pipeline con
        self.preprocessor = DataPreprocessorPipeline()
        self.geometry_1 = EdgeLineCornerPipeline()
        self.geometry_2 = PanoramaStitchingPipeline()
        self.detection = ObjectDetectionPipeline()
        self.segmentation = SegmentationPipeline()
        self.viz = VisualizationPipeline()

        self.last_results = {}

    def run(self, input: Union[str, List[str], np.ndarray], visualize=True) -> Dict[str, Dict[str, Any]]:
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