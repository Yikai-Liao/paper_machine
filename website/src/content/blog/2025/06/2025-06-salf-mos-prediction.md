---
title: "SALF-MOS: Speaker Agnostic Latent Features Downsampled for MOS Prediction"
pubDatetime: 2025-06-02T10:45:40+00:00
slug: "2025-06-salf-mos-prediction"
type: "arxiv"
id: "2506.02082"
score: 0.6129381268733416
author: "grok-3-latest"
authors: ["Saurabh Agrawal", "Raj Gohil", "Gopal Kumar Agrawal", "Vikram C M", "Kushal Verma"]
tags: ["Speech Quality", "MOS Prediction", "Latent Features", "Self-Supervised Learning", "Text-to-Speech"]
institution: ["Samsung R&D Institute Bangalore, India"]
description: "本文提出 SALF-MOS 模型，通过小型端到端架构和潜在特征提取实现高效 MOS 预测，在多个数据集上取得最先进结果，显著降低计算成本和主观评估依赖。"
---

> **Summary:** 本文提出 SALF-MOS 模型，通过小型端到端架构和潜在特征提取实现高效 MOS 预测，在多个数据集上取得最先进结果，显著降低计算成本和主观评估依赖。 

> **Keywords:** Speech Quality, MOS Prediction, Latent Features, Self-Supervised Learning, Text-to-Speech

**Authors:** Saurabh Agrawal, Raj Gohil, Gopal Kumar Agrawal, Vikram C M, Kushal Verma

**Institution(s):** Samsung R&D Institute Bangalore, India


## Problem Background

语音质量评估中的 Mean Opinion Score (MOS) 是评估文本转语音 (TTS) 和语音转换模型的重要主观指标，但传统人工评分方法耗时耗力，且对低资源语言和特定环境具有挑战性；现有的客观指标如 PESQ、POLQA、STOI 与 TTS 质量相关性有限，而神经网络方法在模型参数、跨数据集泛化性和通用性上存在不足，因此需要一种高效、泛化性强的 MOS 预测模型来减少人工依赖并提升评估效率。

## Method

* **核心思想**：提出 SALF-MOS（Speaker Agnostic Latent Features Downsampled for MOS Prediction），一个小型、端到端的模型，通过提取语音的潜在特征预测 MOS 分数，同时强调与说话者无关（Speaker Agnostic），以提高泛化性。
* **特征选择**：以自监督学习 (SSL) 特征（如 wav2vec）作为主要输入，同时实验了 MFCC、LFCC 和 x-vector 等特征，最终发现 wav2vec 表现最佳，因其在大量语音数据上的泛化能力。
* **模型架构**：受 U-Net 启发，包含四层双重卷积 (Double Convolution) 和三层下采样 (Downsampling)；每层双重卷积由 1D 卷积（核大小 3，步长 1，填充 1）、批归一化和 ReLU 激活组成，下采样使用核大小 2 和步长 2 减少维度；提取的特征通过线性层进行潜在特征提取 (LFE)，并堆叠映射到最终线性层输出 MOS 分数。
* **创新点**：不依赖 SSL 模型的预训练或微调，不需要域 ID 或听众 ID，避免多损失函数或多 SSL 模型融合的复杂性，模型参数仅为 1574，计算效率高。
* **目标**：通过特征压缩和下采样实现高效计算，同时保持对不同数据集和语音场景的泛化能力，减少对人工评估的依赖。

## Experiment

* **有效性**：SALF-MOS 在多个数据集（BVCC、VCC2018、SOMOS、TMHINTQI）上取得了最先进 (SOTA) 结果，例如在 BVCC 上 MSE 为 0.144，LCC 为 0.948，SRCC 为 0.946，KTAU 为 0.819，显著优于其他模型（如 MOSNet 的 MSE 0.816，LCC 0.294）。
* **泛化性**：模型在不同分布的数据集上表现稳定，尤其在分布不均的 TMHINTQI 数据集上仍取得较好结果，表明其对多样化语音特征的捕捉能力强。
* **效率**：模型参数极小（1574），远低于其他模型（如 NORESQA-MOS 的 92M），计算成本低，适合实际部署。
* **实验设置合理性**：数据集选择涵盖多种场景，训练-验证-测试数据按 8:1:1 划分，使用早停机制防止过拟合，硬件环境（单 A10 GPU）和超参数（如学习率 1e-4）设置透明；消融实验验证了特征选择（wav2vec 最佳）和模型深度（深度 4 最优）的合理性。
* **结论**：实验设计全面，结果显著，数据支持了 SALF-MOS 在 MOS 预测任务中的优越性和实用性。

## Further Thoughts

SALF-MOS 的 'Speaker Agnostic' 理念启发了我，是否可以在其他语音任务中探索与说话者无关的特征提取方法以提升模型适应性；此外，下采样和特征压缩的思路是否可推广至资源受限场景；模型对多语言或低资源语言的适用性值得进一步研究，或许结合少样本学习或迁移学习能提升表现；另外，是否可以通过设计更轻量级的自监督特征提取模块替代现有 SSL 模型，进一步降低计算成本。