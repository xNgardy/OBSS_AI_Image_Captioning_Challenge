#  OBSS AI Image Captioning Challenge - 2nd Place Solution

**Author:** Pelinsu Kaleli  
**Competition:** OBSS AI Image Captioning Challenge (Kaggle)  
**Rank:** 2nd Place  
**Model:** Qwen2.5-VL-7B-Instruct (Fine-Tuned)

![Rank](https://img.shields.io/badge/Rank-2nd%20Place-gold)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This repository contains the source code and technical report for my submission to the **OBSS AI Image Captioning Challenge**, where I secured **2nd place**.

The goal of the challenge was to generate high-quality, descriptive captions for a diverse dataset of images. My solution leverages **Vision-Language Models (VLMs)**, specifically fine-tuning the **Qwen2.5-VL-7B-Instruct** model using **Low-Rank Adaptation (LoRA)** to achieve state-of-the-art performance with efficient memory usage.

##  Approach & Methodology

### 1. Model Architecture
* **Base Model:** `Qwen2.5-VL-7B-Instruct` (A powerful multimodal transformer).
* **Optimization:** Implemented **4-bit quantization** using `BitsAndBytes` to fit the 7B parameter model into GPU memory constraints (Google Colab A100/T4).
* **Fine-Tuning:** Used **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA** to adapt the model to the specific captioning style of the competition dataset without retraining all parameters.

### 2. Training Pipeline
* **Library:** `TRL` (Transformer Reinforcement Learning) library for supervised fine-tuning (`SFTTrainer`).
* **Prompt Engineering:** Designed a specific system prompt to encourage "comprehensive, objective descriptions focusing on multiple visual elements".
* **Precision:** Mixed precision training (BF16) where supported.

## Tech Stack

* **Deep Learning:** `PyTorch`, `Transformers` (Hugging Face).
* **VLM Utilities:** `qwen_vl_utils`.
* **Optimization:** `bitsandbytes`, `peft` (LoRA).
* **Data Processing:** `pandas`, `PIL` (Python Imaging Library).

## ⚙️ Installation

To reproduce the environment, install the required dependencies:

```bash
pip install torch transformers qwen_vl_utils
pip install -U bitsandbytes
pip install trl==0.12.0 peft pandas pillow
