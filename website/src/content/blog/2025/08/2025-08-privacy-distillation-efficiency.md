---
title: "Improving Noise Efficiency in Privacy-preserving Dataset Distillation"
pubDatetime: 2025-08-03T13:15:52+00:00
slug: "2025-08-privacy-distillation-efficiency"
type: "arxiv"
id: "2508.01749"
score: 0.3323438319866359
author: "grok-3-latest"
authors: ["Runkai Zheng", "Vishnu Asutosh Dasu", "Yinong Oliver Wang", "Haohan Wang", "Fernando De la Torre"]
tags: ["Dataset Distillation", "Differential Privacy", "Privacy Preservation", "Signal Optimization", "Synthetic Data"]
institution: ["Carnegie Mellon University", "Pennsylvania State University", "University of Illinois Urbana-Champaign"]
description: "本文提出 Dosser 框架，通过解耦采样与优化及子空间误差减少策略，在隐私保护数据集蒸馏中显著提高噪声效率和合成数据集实用性。"
---

> **Summary:** 本文提出 Dosser 框架，通过解耦采样与优化及子空间误差减少策略，在隐私保护数据集蒸馏中显著提高噪声效率和合成数据集实用性。 

> **Keywords:** Dataset Distillation, Differential Privacy, Privacy Preservation, Signal Optimization, Synthetic Data

**Authors:** Runkai Zheng, Vishnu Asutosh Dasu, Yinong Oliver Wang, Haohan Wang, Fernando De la Torre

**Institution(s):** Carnegie Mellon University, Pennsylvania State University, University of Illinois Urbana-Champaign


## Problem Background

现代机器学习依赖大规模数据集，但这些数据常包含敏感信息，存在隐私泄露风险。
差分隐私（Differential Privacy, DP）通过生成合成数据集限制隐私泄露，但需要大量数据以接近原始数据性能，导致计算和存储成本高昂。
数据集蒸馏（Dataset Distillation, DD）通过生成少量高信息量合成样本降低存储需求，然而现有结合 DP 的 DD 方法由于采样与优化耦合及依赖随机初始化网络提取信号，导致噪声利用效率低，影响合成数据集实用性。
本文旨在解决如何在严格隐私预算下提高噪声效率，生成紧凑且有用的合成数据集，同时保持隐私保护。

## Method

*   **核心思想:** 提出名为 Dosser 的框架，通过解耦采样与优化过程并提升信号质量，在隐私预算内提高噪声效率，增强合成数据集的实用性。
*   **解耦优化与采样（Decoupled Optimization and Sampling, DOS）:** 
    *   将训练信号采样和合成数据集优化过程分开，采样阶段从私有数据提取 DP 保护的训练信号并存储，优化阶段基于这些信号迭代更新合成数据集。
    *   这种解耦允许在固定隐私预算下减少采样次数（降低噪声累积），同时增加优化迭代次数以提高合成数据集质量。
    *   具体实现上，采样阶段包括数据采样、信号提取（特征或梯度）、裁剪以限制敏感性、聚合并添加高斯噪声以满足 DP 要求；优化阶段通过最小化合成信号与存储的噪声信号之间的距离（如 L2 距离）更新合成数据集。
*   **子空间误差减少（Subspace-based Error Reduction, SER）:** 
    *   利用辅助数据集（通过预训练生成模型如 Stable Diffusion 或 DP 训练的生成模型生成）识别随机初始化网络中的信息子空间。
    *   将训练信号投影到该子空间，集中信号功率于高实用性维度，提高信噪比（SNR），从而减轻 DP 噪声的影响。
    *   理论分析表明，子空间投影通过维度减少和丢弃无信息方差降低均值估计的均方误差（MSE），并通过辅助数据集避免额外隐私成本。
*   **关键优势:** 不增加隐私预算，通过采样解耦减少噪声注入，通过子空间投影提升信号质量，两者结合显著提升合成数据集性能。

## Experiment

*   **有效性:** 实验在 MNIST、FashionMNIST 和 CIFAR-10 数据集上进行，隐私预算为 (ε=1, δ=10^-5)，Dosser 相比现有方法（如 NDPDC）在准确率上显著提升，尤其在 CIFAR-10 上，IPC=10 时提升 10.6%，IPC=50 时提升 10.0%，且与无 DP 约束的基准方法性能差距缩小（如 CIFAR-10 IPC=10 时仅差 1.5%）。
*   **模块贡献:** 消融研究表明，DOS 单独提升准确率约 4.3%，SER 单独提升约 1.9%，两者结合提升 5.7%，证明两者互补性，DOS 提供更多优化空间，SER 增强信号质量。
*   **参数影响:** 实验验证了解耦优化迭代次数增加带来的性能提升（尤其在高 IPC 下），以及 SER 中子空间维度和辅助数据集大小的影响，SER 在高噪声场景下降噪效果显著。
*   **合理性:** 实验设置全面，涵盖不同数据集、隐私预算和超参数配置（如 IPC=10 和 50，优化迭代从 50k 到 200k），对比了多种基线方法（如 DP-Sinkhorn、PSG、NDPDC），数据支持结论合理。

## Further Thoughts

解耦采样与优化的策略可推广至其他隐私保护任务，如联邦学习，通过减少噪声注入次数提升性能；SER 的子空间投影方法启发在信号处理中引入辅助数据或预训练模型增强信噪比，尤其在隐私预算紧张时作为通用降噪手段；对于特定领域数据集分布不匹配问题，未来可探索自适应子空间学习或跨领域迁移方法以提升 SER 适用性。