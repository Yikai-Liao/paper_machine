---
title: "Energy-based Preference Optimization for Test-time Adaptation"
pubDatetime: 2025-05-26T07:21:32+00:00
slug: "2025-05-energy-preference-tta"
type: "arxiv"
id: "2505.19607"
score: 0.6255757332681958
author: "grok-3-latest"
authors: ["Yewon Han", "Seoyun Yang", "Taesup Kim"]
tags: ["Energy Model", "Test-Time Adaptation", "Preference Optimization", "Calibration", "Distribution Shift"]
institution: ["Seoul National University"]
description: "本文提出 E PO TTA，一种基于能量模型和偏好优化的测试时适应方法，通过无采样策略显著提升准确性和校准性能，同时大幅降低计算成本。"
---

> **Summary:** 本文提出 E PO TTA，一种基于能量模型和偏好优化的测试时适应方法，通过无采样策略显著提升准确性和校准性能，同时大幅降低计算成本。 

> **Keywords:** Energy Model, Test-Time Adaptation, Preference Optimization, Calibration, Distribution Shift

**Authors:** Yewon Han, Seoyun Yang, Taesup Kim

**Institution(s):** Seoul National University


## Problem Background

测试时适应（Test-Time Adaptation, TTA）旨在提高模型在测试数据分布与训练数据分布不一致时的鲁棒性，但现有方法如熵最小化依赖于不确定的模型预测，导致过自信和校准误差增加；而基于能量的模型虽能通过边缘分布建模避免预测依赖，但计算成本高昂（如需大量 SGLD 采样），不适合实时场景。

## Method

*   **核心思想:** 提出 E PO TTA（Energy-based Preference Optimization for Test-time Adaptation），结合能量模型和直接偏好优化（Direct Preference Optimization, DPO）框架，通过偏好对优化目标分布，无需采样或显式计算归一化常数。
*   **具体实现:** 
    *   将目标分布参数化为源分布与残差能量函数的组合，捕捉分布偏移。
    *   利用 DPO 的数学等价性，假设目标数据优于源数据，通过源数据和目标数据的偏好对，构建负对数似然优化目标，直接调整模型适应目标分布。
    *   将源模型和目标模型均表示为能量函数形式，通过代数简化消除归一化常数的计算需求。
    *   使用小型源数据缓冲区和目标数据批次构建偏好对，优化过程中动态调整不确定样本的梯度贡献（通过能量与梯度系数的负相关性）。
*   **关键优势:** 避免了熵最小化导致的过自信问题，同时克服了传统能量模型的高计算开销，适合实时适应场景。

## Experiment

*   **有效性:** E PO TTA 在 CIFAR10-C、CIFAR100-C 和 TinyImageNet-C 数据集上均表现出色，特别是在 TinyImageNet-C 最高损坏严重程度（Level 5）下，准确率达 40.30%，优于 TEA（39.96%）和 TENT（39.83%）；校准误差（ECE）也显著降低，如 TinyImageNet-C 上平均 ECE 为 11.85%，优于大多数基线。
*   **效率:** 计算成本大幅降低，GFLOPs 仅为 527.40，远低于 TEA 的 4335.82，适合实时 TTA 场景。
*   **鲁棒性:** 在非独立同分布（Non-IID）场景下，平均准确率比 TENT 高 2.14 个百分点；消融实验表明对源数据缓冲区大小和内容不敏感，仅用 1% 源数据仍维持性能。
*   **实验设置:** 覆盖多种损坏类型和严重程度，数据集选择合理，评价指标包括准确率、校准误差和计算成本，实验设计全面。

## Further Thoughts

E PO TTA 的能量模型与偏好优化结合的思路启发我们可以在无监督域适应或在线学习中探索类似框架，利用偏好对简化复杂分布建模；此外，通过数学等价性消除归一化常数计算的技巧可推广至其他概率模型，而不确定性样本的梯度重加权机制或可应用于噪声标签学习，提升模型鲁棒性。