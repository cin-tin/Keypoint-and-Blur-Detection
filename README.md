# Keypoint-and-Blur-Detection
This repository contains scripts for detecting human keypoints (hands, pose, and face) in videos and classifying frames based on the presence of these keypoints. The project also identifies blurred frames and explores various Mediapipe libraries to experiment with detection capabilities under different conditions.

---

## Purpose

This project aims to:
- **Identify Blur Frames**: Detect frames where keypoints are missing or blurred.
- **Experiment with Mediapipe Libraries**: Test and analyze Mediapipe's hands, pose, and holistic solutions for accuracy and robustness.
- **Frame Classification**: Categorize video frames based on the completeness of detected keypoints.

---

## Features

- **Keypoint Detection**:
  - Uses Mediapipe to extract keypoints for hands, pose, and face.
  - Visualizes keypoints by overlaying them on the frames.
- **Frame Classification**:
  - Frames are grouped into categories:
    - Fully detected keypoints.
    - Missing or blurred keypoints (e.g., left hand, right hand, pose, or face).
- **Blur Analysis**:
  - Detects and categorizes blur frames by analyzing keypoint absence.
- **Batch Processing**:
  - Supports processing multiple videos efficiently.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Folder Structure](#folder-structure)
4. [Examples](#examples)
5. [Contributing](#contributing)
6. [License](#license)

---

## Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/keypoint-detection-classification.git
cd Keypoint-and-Blur-Detection
