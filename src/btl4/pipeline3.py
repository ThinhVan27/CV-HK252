import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Optional, Tuple

from base_pipeline import *

class ObjectDetectionPipeline(BasePipeline):
    """
    Object Detection Pipeline
    """
    def run(self, input: Union[str, List[str], List[np.ndarray]]) -> Dict[str, Any]:
        """
        Run `OjectDetectionPipeline`.
        
        Args:
            @input: either image path, list of image paths or image tensor.
        """
        pass