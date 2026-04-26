# Digital Image Processing and Computer Vision Assignments - Semester 252
This repository includes four assignments in Digital Image Processing and Computer Vision Course at Ho Chi Minh University of Technology - VNUHCM, semester 252.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

## Overview

| Assignment | Content | Colab Notebook |
|:---:|---|---|
| 1 | - Illustrate digital images via gray and RGB channels, analysize and evaluate the effect of each color channels on image representation.<br> - Implement low-pass filter (**Box Filter** and **Gaussian Filter**), using them for image smoothing and denoising. <br> - Apply high-pass filter (**Sobel Filter** and **Laplacian Filter**) to detect edges and sharpening images. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zdEs-BdALYo9ZlPimKYUhOgAcnWtZzzH?usp=sharing)|
| 2 | - Apply basic and advanced geometric transformations in images. <br> - Implement **Alpha Blending** - intensity-based blending - and **Poisson Blending** - gradient-based blending - to blend an object into a background image, assuring seamless and plausible boundary regions.| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1462S78DcQIz1u6G3vBrfEY8es-Q8bptI?usp=sharing)|
| 3 | - Implement a feature-based image stitching pipeline for panoramic image generation. <br> - Compare multiple local feature extraction methods (SIFT, ORB, PCA-SIFT/SURF alias) in keypoint detection and matching quality. <br> - Estimate homography with robust matching (RANSAC), then blend aligned images to produce final panoramas. <br> - Evaluate stitching quality through matching/alignment metrics and visual quality metrics (MSE, PSNR, SSIM). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OvNvGOTwjwfB4xWPlBIcyBM1pWVBEIEO?usp=sharing)|
| 4 | - Design and implement a Computer Vision pipeline end-to-end, including *Preprocessing, Geometry Feature Detection, Image Stitching, Pedestrian Detection* and *Image Segmentation*. <br> - Evaluate the quantitative and qualitative each modules.| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19sbtFykD_7Vq7ZLRluJNZ3_8v2r60JXG?usp=sharing)|

## Prerequisites
- Python 3.12 or higher
- Experience in programming with Python
- Have fundamental knowledge about Calculus/ Linear Algebra/ Probability & Statistics and general Mathematics.
- Familiar with basic Machine Learning/ Deep Learning

## Quick Start
1. Clone the repository
```
git clone https://github.com/ThinhVan27/CV-HK252.git
cd CV-HK252
```
2. Create environment
```
python -m venv venv
```
3. Activate environment
```
# On MacOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate.bat
```
4. Install dependencies
```
pip install -r requirements.txt
```
## Project Structure
```text
CV-HK252/
├── LICENSE
├── README.md
├── requirements.txt
├── yolo26n-seg.pt
├── yolov8n.pt
├── checkpoint/
│   └── superpoint_no_borders.onnx
├── docs/
│   ├── btl1.html
│   ├── btl2.html
│   ├── btl3.html
│   ├── btl4.html
│   ├── index.html
│   ├── README.md
│   └── static/
│       ├── css/
│       │   ├── bulma-carousel.min.css
│       │   ├── bulma-slider.min.css
│       │   ├── bulma.css.map.txt
│       │   ├── bulma.min.css
│       │   ├── fontawesome.all.min.css
│       │   └── index.css
│       ├── images/
│       │   └── btl1/
│       ├── js/
│       │   ├── bulma-carousel.js
│       │   ├── bulma-carousel.min.js
│       │   ├── bulma-slider.js
│       │   ├── bulma-slider.min.js
│       │   ├── fontawesome.all.min.js
│       │   └── index.js
│       ├── pdfs/
│       └── videos/
├── img/
│   ├── btl1/
│   ├── btl2/
│   ├── btl3/
│   │   ├── BK/
│   │   ├── Cafe/
│   │   ├── Desk/
│   │   └── Lab/
│   └── btl4/
│       ├── Detection/
│       ├── GeometryFeature/
│       ├── Overall/
│       └── Segment/
│           ├── img/
│           └── mask/
└── src/
	├── btl1/
	│   └── pipeline.py
	├── btl2/
	│   ├── pipeline2.py
	│   └── pipeline3.py
	├── btl3/
	│   ├── evaluation.py
	│   └── pipeline.py
	└── btl4/
		├── base_pipeline.py
		├── overall_pipeline.py
		├── pipeline1.py
		├── pipeline2.py
		├── pipeline3.py
		├── pipeline4.py
		└── requirements.txt
```

## License
This project has implemented for academic purpose only.
