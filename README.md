# RiceDoctor-MobileNetV4

# RiceLeaf Disease Classification using MobileNetV4

## Overview

This project implements a deep learning pipeline for classifying rice leaf diseases from leaf images using a MobileNetV4-based convolutional neural network. The system is designed to distinguish between common rice leaf conditions, including bacterial leaf blight, brown spot, rice blast, and healthy leaves.

The repository focuses on building a reproducible training workflow for image classification using PyTorch and TorchVision.

## Problem Statement

Rice diseases can cause significant yield loss if they are not detected early. Manual inspection of plant health is time-consuming and often inconsistent. Automated image-based diagnosis can help identify disease patterns earlier and support agricultural monitoring efforts.

This project explores a lightweight computer vision approach for disease recognition using pretrained MobileNetV4 features and a custom classifier head.

## Objectives

- Build a clean image classification pipeline for rice leaf disease detection
- Train a MobileNetV4 model on labeled plant disease image datasets
- Evaluate model performance using validation metrics
- Provide a reproducible training structure for future experimentation
- Create a well-organized repository suitable for academic or research-oriented project submissions

## Dataset

The project expects a structured dataset organized into train and validation folders by class.

```text
data/
└── rice_leaf_diseases/
    ├── train/
    │   ├── Bacterial_Leaf_Blight/
    │   ├── Brown_Spot/
    │   ├── Healthy/
    │   └── Rice_Blast/
    └── val/
        ├── Bacterial_Leaf_Blight/
        ├── Brown_Spot/
        ├── Healthy/
        └── Rice_Blast/
