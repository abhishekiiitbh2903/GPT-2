# GPT-2 Implementation from Scratch Using PyTorch

This repository contains an implementation of the GPT-2 model from scratch using PyTorch. The model is trained on the FineWeb EDU dataset, consisting of 10 billion tokens, and is evaluated using the HellaEval evaluation method, where it outperformed the GPT-2 benchmark.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Training](#training)
- [Optimizations](#optimizations)
- [HellaEval Performance Comparison](#hellaeval-performance-comparison)
- [Possible Reasons for Improved Performance](#possible-reasons-for-improved-performance)

## Features
- Implementation of GPT-2 architecture from scratch
- Evaluation using the HellaEval method
- Performance surpassing the GPT-2 benchmark
- Efficient GPU usage with optimization strategies

## Dataset
The model is trained on the **FineWeb EDU dataset**, which consists of educational content. The dataset is available on [Hugging Face](https://huggingface.co/datasets).

## Training
The model was trained on **10 billion tokens** from the FineWeb EDU dataset. The training procedure involves standard language modeling tasks where the model predicts the next token in a sequence.

## Optimizations
To improve GPU efficiency during training, the following optimizations were implemented:
- Utilized numbers that are multiples of 2, as GPU cores operate more efficiently in powers of 2. This helps in reducing round trip time.

## HellaEval Performance Comparison
![HellaEval Performance Comparison](path_to_your_image.png)  


## Possible Reasons for Improved Performance
1. **Potential Data Leakage**: It is uncertain whether the FineWeb EDU dataset contains data from HellaEval, which could lead to data leakage and better performance.
2. **Focused Dataset**: Unlike GPT-2, which was trained on a diverse dataset, my model was primarily trained on educational content, allowing it to generalize better within that domain.
