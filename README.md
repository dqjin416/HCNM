# HCNM: Hierarchical Cognitive Neural Model for Small-Sample Image Classification

## Overview
This repository implements the **Hierarchical Cognitive Neural Model (HCNM)** proposed in the paper:  
**"HCNM: Hierarchical cognitive neural model for small-sample image classification"** (Expert Systems with Applications, 2025).  
📄 [Paper Link](https://www.sciencedirect.com/science/article/pii/S0957417425005263)

HCNM is a biologically inspired model for small-sample image classification, simulating human visual cognition through hierarchical neural fields (V4 and IT) and a pre-trained Vision Transformer (ViT) for feature extraction. It achieves state-of-the-art performance on multiple datasets with minimal parameter tuning.

---

## Key Features
- **Biologically Plausible**: Simulates visual cortex mechanisms (V1-V2-V4-IT hierarchy).  
- **Efficient SSL**: Optimized for small-sample learning via Representative Point Neurons (RPNs) and adaptive lateral interactions.  
- **Interpretable**: Neural field dynamics provide transparent decision-making.  
- **High Accuracy**: Outperforms existing methods on LabelMe, UIUC-Sports, 15Scenes, and BMW-10 datasets.

---

## Datasets Supported

| Dataset         | Classes | Samples   | Use Case                                  |
|-----------------|---------|-----------|-------------------------------------------|
| LabelMe         | 8       | 2,688     | Natural scene classification              |
| UIUC-Sports     | 8       | 1,579     | Sports scene recognition                  |
| 15Scenes        | 15      | 4,485     | Comprehensive scene classification       |
| BMW-10          | 10      | 512       | Fine-grained vehicle classification       |
| CUB-200-2011    | 200     | 11,788    | Few-shot fine-grained bird classification |
| Stanford-Dogs   | 120     | 20,580    | Few-shot dog breed classification         |
| FC-100          | 100     | 60,000    | Few-shot learning benchmark              |
| CIFAR10         | 10      | 60,000    | General object recognition               |
| miniImageNet    | 100     | 60,000    | Few-shot learning benchmark              |

### Notes:
- **LabelMe**, **UIUC-Sports**, and **15Scenes** were used for small-sample image classification experiments.  
- **BMW-10** was tested with a 70-30 train-test split for fine-grained classification.  
- **CUB-200-2011**, **Stanford-Dogs**, **FC-100**, and **miniImageNet** were evaluated under 5-way 1-shot and 5-way 5-shot settings for few-shot learning.  
- **CIFAR10** served as a baseline dataset for broader applicability validation.  

## Installation
```bash
git clone https://github.com/dqjin416/HCNM.git
cd HCNM
conda env create --file environment.yml  # Install dependencies (PyTorch, NumPy, SciPy)
conda activate HCNM
```

## Citation
```sh
@article{JIN2025126904,
title = {HCNM: Hierarchical cognitive neural model for small-sample image classification},
journal = {Expert Systems with Applications},
volume = {276},
pages = {126904},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.126904},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425005263},
author = {Dequan Jin and Ruoge Li and Nan Xiang and Di Zhao and Xuanlu Xiang and Shihui Ying},
keywords = {Visual classification, Small-sample learning, Neural field, Hierarchical architecture},
abstract = {Small-sample image classification is a hot topic in computer vision. Despite the progress made by some deep neural networks in solving the small-sample learning problem, there remain challenges in learning efficiently and robustly. These challenges can affect the overall performance and effectiveness of the model. To address these issues, we propose a hierarchical cognitive neural model (HCNM) based on the simulation of visual cognition to construct the sparse structure of the neural model from the perspective of semi-supervised learning. We use a deep learning network for feature extraction and two coupled dynamic neural field equations to simulate the encoding and classification functions in visual image recognition and classification. The model simulates macroscopic neural activation in object recognition and identifies representative point neurons (RPNs) by evaluating the magnitude of lateral interactions within the V4 neural field on an adaptive cognitive scale. Our approach provides an efficient small-sample image classification algorithm that does not require complex parameter tuning and maintains biological plausibility and interpretability. Experimental results using four real-world image datasets demonstrate the superior performance of our model and method for small-sample image classification compared to other state-of-the-art research methods.}
}
```
