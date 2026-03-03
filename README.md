# ♻️ Multiclass Semantic Segmentation — Waste Sorting Optimization

Pixel-level waste classification using deep learning to automate and improve recycling sorting processes.

## Overview

Manual waste sorting faces significant challenges: fast and complex material flows, human visual limitations, and environmental consequences from misclassification. This project addresses these issues through **multiclass semantic segmentation**, enabling precise, pixel-level identification of waste materials on conveyor belts.

## Model Architecture

**U-Net** with a **ResNet34** encoder (Transfer Learning)

- Encoder–decoder architecture with skip connections
- Output: multiclass segmentation masks
- Classes: **Plastic**, **Paper**, **Metal**, **Other** (background)

## Dataset

**TACO** *(Trash Annotations in Context)*

- 1,500 images in COCO format
- Open-source with segmentation annotations
- 60 original categories regrouped into 4 super-categories
- High image quality and variability

## Data Preprocessing

- 60 original classes merged into 4 super-categories
- COCO annotations converted to `.png` mask images
- Final mask format: pixel values from 0 to 4
- Training performed on Google Colab

## Results

| Metric | Value |
|---|---|
| Validation Loss | 0.24 |
| Validation Dice Score | 0.58 |

## Future Improvements

- Rebalance the dataset (increase plastic samples)
- Apply data augmentation (angle and lighting variations)
- Address category confusion from TACO's highly specific sub-classes

## Team

Fernando Lasmar · Yan Ding · Flávio Rosim · Matheus Galdino
