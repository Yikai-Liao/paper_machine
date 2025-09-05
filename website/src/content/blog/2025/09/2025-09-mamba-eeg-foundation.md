---
title: "Mentality: A Mamba-based Approach towards Foundation Models for EEG"
pubDatetime: 2025-09-02T18:47:38+00:00
slug: "2025-09-mamba-eeg-foundation"
type: "arxiv"
id: "2509.02746"
score: 0.46200365218640804
author: "grok-3-latest"
authors: ["Saarang Panchavati", "Corey Arnold", "William Speier"]
tags: ["EEG Analysis", "Foundation Model", "Sequence Modeling", "State Space Model", "Seizure Detection"]
institution: ["University of California Los Angeles"]
description: "本文通过基于 Mamba 的选择性状态空间模型，结合自监督预训练和下游癫痫检测任务，初步展示了构建 EEG 基础模型的潜力，为神经数据分析提供了新思路。"
---

> **Summary:** 本文通过基于 Mamba 的选择性状态空间模型，结合自监督预训练和下游癫痫检测任务，初步展示了构建 EEG 基础模型的潜力，为神经数据分析提供了新思路。 

> **Keywords:** EEG Analysis, Foundation Model, Sequence Modeling, State Space Model, Seizure Detection

**Authors:** Saarang Panchavati, Corey Arnold, William Speier

**Institution(s):** University of California Los Angeles


## Problem Background

脑电图（EEG）作为诊断神经疾病（如癫痫）的关键工具，因其信号噪声大、高维、非线性及个体差异显著而难以分析，传统机器学习方法无法有效捕捉其复杂的时空动态，且临床上依赖耗时的人工检查；本文探索基于深度学习的序列建模技术，特别是基础模型（foundation models），以提升 EEG 分析的自动化和泛化能力。

## Method

* **核心思想**：利用 Mamba 选择性状态空间模型（selective state space model）构建 EEG 基础模型，通过捕捉长序列依赖和时序动态，提升对复杂 EEG 信号的建模能力。
* **数据处理**：基于 Temple University Hospital EEG Seizure Corpus (TUSZ) 数据集，包含癫痫发作和非发作记录，信号重采样至 200 Hz，移除 60/120 Hz 干扰，分段为 10 秒窗口。
* **模型架构**：设计编码器-解码器结构，初始层采用 1D CNN（核大小 100）学习频率滤波器（至 50 Hz），通过线性层进行通道混合，随后多个 Mamba 块（包含 Mamba 层、层归一化和残差连接）学习时序动态；采用 U-Net 风格的下采样（双卷积 + 均值池化）和上采样（Mamba-转置卷积-双卷积）结构，结合残差连接保留特征；下游任务中，通过最大池化和线性层输出分类概率。
* **训练策略**：首先通过自监督重建任务预训练模型，结合均方误差（MSE）和频谱损失（spectral loss）优化信号重建质量；随后在下游癫痫检测任务中微调模型。
* **关键点**：Mamba 块高效处理长序列依赖，预训练提升泛化能力，频谱损失增强对信号高频成分的捕捉。

## Experiment

* **预训练效果**：最佳模型在重建任务上达到 MSE 0.0063，加入频谱损失后性能显著优于无频谱损失（MSE 0.025），重建示例显示模型能捕捉信号整体模式，但对高频成分仍有不足。
* **下游任务效果**：预训练模型在癫痫检测任务中取得 AUROC 0.72，相比从头训练的模型（AUROC 0.64）有明显提升，证明自监督预训练的有效性。
* **实验设置**：数据分割合理（训练集 579 名患者，测试集 43 名患者），确保患者间独立性，避免数据泄露；但 AUROC 0.72 距临床应用仍有差距，可能受限于模型对空间关系的建模不足。
* **解释性分析**：通过初始卷积层权重和通道显著性图（saliency map）探究模型关注点，例如癫痫样本中 T4 和 P4 通道被高亮，符合癫痫高频信号特征。
* **计算开销**：未详细提及，但 Mamba 模型对长序列的高效处理可能使其在实际应用中具有优势。

## Further Thoughts

论文提出未来可通过图神经网络（GNN）建模 EEG 通道间的空间关系，启发我思考是否能设计动态图结构，根据时间窗口自适应调整连接权重；此外，随机掩码通道的鲁棒性训练策略让我联想到可引入更多数据增强技术（如模拟硬件噪声）以提升跨场景泛化能力；最后，显式利用 Mamba 的状态空间动态结合行为数据建模，可能揭示神经-行为关系的深层机制，不仅限于 EEG，也可推广至其他生物信号分析。