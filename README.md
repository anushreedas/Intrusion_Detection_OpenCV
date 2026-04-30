# Intrusion Detection System (Computer Vision | OpenCV)

A computer vision project that detects and segments intruders from image sequences using classical techniques in OpenCV.
This project demonstrates strong fundamentals in image processing, segmentation, and algorithm design without relying on deep learning.

# Problem Statement

Detect and segment a moving intruder from a sequence of images captured by a static surveillance camera, while minimizing noise and false detections caused by lighting changes or background variations.

# Solution Overview

The system combines multiple classical CV techniques into a single pipeline:

1. Background Modeling - Uses MOG2 (Mixture of Gaussians) to learn scene dynamics.
2. Foreground Extraction - Identifies moving regions as potential intruders.
3. Noise Reduction - Applies morphological operations and component filtering.
4. Segmentation - Uses the watershed algorithm to precisely isolate the intruder.
5. Visualization - Outputs mosaics of segmented frames for easy inspection.

# How to Run
```
git clone https://github.com/anushreedas/Intrusion_Detection_OpenCV.git
cd Intrusion_Detection_OpenCV
pip install opencv-python numpy
python IntrusionDetection.py
```

# Tech Stack
Language: Python

Libraries:
* OpenCV
* NumPy

Concepts:
* Background Subtraction (MOG2)
* Morphological Image Processing
* Connected Component Analysis
* Watershed Segmentation

# Limitations
* Assumes static camera setup
* Sensitive to significant illumination changes
* Not optimized for real-time streaming
* Struggles with multiple overlapping objects

# Future Improvements
* Integrate deep learning models (e.g., Mask R-CNN, YOLO segmentation)
* Add real-time video stream processing
* Implement object tracking (e.g., SORT / DeepSORT)
* Improve illumination normalization
* Deploy as a web-based monitoring tool
