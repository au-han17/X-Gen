# 🩻 X-Gen: Enhancing Radiology Report Generation via LLM-Driven Data Augmentation and Decoupled Training

[![Conference](https://img.shields.io/badge/DICTA-2025%20Oral-red.svg)](https://dicta2025.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation of the **DICTA 2025 Oral Presentation**:  
**"X-Gen: Enhancing Radiology Report Generation via LLM-Driven Data Augmentation and Decoupled Training"**

> **Note:** This repository currently focuses on the implementation of the **IU X-ray** dataset using **R2Gen** as the baseline report generation branch.

---

## 📌 Abstract

The scarcity and limited accessibility of medical data significantly challenge deep learning applications in medical AI. Radiology report generation (RRG), a key medical AI research area, could greatly improve computer-aided diagnosis through automated X-ray image interpretation. However, obtaining paired X-ray images and reports is labor-intensive and restricted by strict regulations. 

Large language models (LLMs), such as GPT-4, provide a promising alternative by enabling cost-effective text data augmentation and report rewriting in varied styles. We rigorously assess augmented data's clinical accuracy and stylistic similarity to radiologist-authored reports through expert evaluations. Interestingly, augmented data enhances RRG model performance, yet performance declines when augmented data surpasses original data volume due to **style distribution shifts**. 

To mitigate this, we propose integrating a **conditional variational autoencoder (cVAE)** into the RRG model to separate medical semantics from writing styles during training, enabling better handling of augmented data's distribution shift. Our proposed method, **X-Gen**, combines data augmentation with decoupled training. Tested on two public Chest X-ray datasets and a private abdomen X-ray dataset, X-Gen significantly improves the performance of baseline models, showcasing its effectiveness and versatility in X-ray report generation.

---

## 📐 Architecture & Data Flow

Below is the overarching dataflow and decoupling pipeline of the **X-Gen** framework:

<p align="center">
  <img src="assets/dataflow.png" alt="X-Gen Dataflow" width="90%"/>
</p>

---

## 🚀 Quick Start

### 1. Environment Setup

Clone the repository and create a conda environment:

```bash
git clone https://github.com/uakhan17/X-Gen.git
cd X-Gen

# Create and activate environment
conda create -n xgen python=3.8 -y
conda activate xgen

# Install dependencies
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install transformers scipy opencv-python
```

---

## 📊 Data Preparation

### Step 1: Download Official IU X-Ray Images
1. Download the official image dataset directly from https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university.
2. We have provided augmented IU X-ray reports rewritten in three different styles and the original ones in * **`data/iu_xray_aug3.json`**.

### Step 2: LLM-Driven Report Augmentation
To generate augmented report variants using OpenAI's API, run the provided augmentation script:

```bash
python chatgpt_submission_public.py \
```
---

## 🏋️ Training & Evaluation

To train the **R2Gen + cVAE Decoupled Training** architecture on the IU X-ray dataset:

```bash
# Train the X-Gen model with decoupled training
bash run_iu_xray.sh
```

## 📝 Citation

If you find **X-Gen** useful for your research or applications, please cite our paper:

```bibtex
@inproceedings{xgen2025dicta,
  title     = {X-Gen: Enhancing Radiology Report Generation via LLM-Driven Data Augmentation and Decoupled Training},
  author    = {Your Name and Co-authors},
  booktitle = {2025 International Conference on Digital Image Computing: Techniques and Applications (DICTA)},
  year      = {2025}
}
```
---

## 🙏 Acknowledgements

This repository builds upon and adapts components from [R2Gen](https://github.com/cqy2019/R2Gen). We express our gratitude to the authors for open-sourcing their work.