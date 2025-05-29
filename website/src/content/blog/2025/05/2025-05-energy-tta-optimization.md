---
title: "Energy-based Preference Optimization for Test-time Adaptation"
pubDatetime: 2025-05-26T07:21:32+00:00
slug: "2025-05-energy-tta-optimization"
type: "arxiv"
id: "2505.19607"
score: 0.6255757332681958
author: "grok-3-latest"
authors: ["Yewon Han", "Seoyun Yang", "Taesup Kim"]
tags: ["Energy-Based Model", "Test-Time Adaptation", "Preference Optimization", "Distribution Shift", "Calibration"]
institution: ["Seoul National University"]
description: "本文提出 E PO TTA，一种无需采样的基于能量的测试时适应方法，通过偏好优化框架显著提升模型在分布偏移下的准确性和校准性能，同时大幅降低计算成本。"
---

> **Summary:** 本文提出 E PO TTA，一种无需采样的基于能量的测试时适应方法，通过偏好优化框架显著提升模型在分布偏移下的准确性和校准性能，同时大幅降低计算成本。 

> **Keywords:** Energy-Based Model, Test-Time Adaptation, Preference Optimization, Distribution Shift, Calibration

**Authors:** Yewon Han, Seoyun Yang, Taesup Kim

**Institution(s):** Seoul National University


## Problem Background

测试时适应（Test-Time Adaptation, TTA）旨在提高深度学习模型在测试数据分布与训练分布不一致时的鲁棒性和泛化能力。
传统 TTA 方法依赖模型自身的预测结果作为伪标签或优化目标，但由于缺乏真实标签，这些预测往往不可靠，导致过自信预测和校准误差增加。
基于能量的模型（Energy-Based Models, EBMs）通过直接对目标数据的边缘分布建模提供了一种替代方案，但现有方法需要大量采样来近似归一化常数，计算成本高昂，不适合实时适应场景。

## Method

*   **核心思想:** 提出 E PO TTA（Energy-based Preference Optimization for Test-time Adaptation），一种无需采样的基于能量的测试时适应方法，通过残差能量函数和直接偏好优化（Direct Preference Optimization, DPO）框架，高效适应目标分布。
*   **具体实现:** 
    *   将目标分布参数化为源分布与残差能量函数的指数形式乘积，捕捉分布偏移。
    *   利用 DPO 的数学等价性，通过目标样本与源样本的偏好对（preference pairs）直接优化目标模型，而无需显式训练残差能量函数或计算归一化常数。
    *   将源模型和目标模型重新定义为能量函数形式，通过代数简化消除归一化常数的影响，确保计算效率。
    *   在优化过程中，使用包含少量源数据的小型回放缓冲区（replay buffer），并通过不确定性感知的梯度加权机制，优先考虑低能量（高置信度）样本，稳定适应过程。
*   **关键优势:** 避免了传统能量方法中高成本的采样过程，同时通过偏好优化提升了模型校准和准确性，适用于实时 TTA 场景。

## Experiment

*   **有效性:** E PO TTA 在 CIFAR10-C、CIFAR100-C 和 TinyImageNet-C 数据集上均优于基线方法（如 TENT、TEA），在 TinyImageNet-C 最高严重度（level 5）上准确率达 40.30%，显著高于其他方法。
*   **校准性能:** 其预期校准误差（ECE）在 TinyImageNet-C 上为 11.85%，优于大多数基线，表明预测置信度与实际准确性更一致，相比 TEA 在复杂数据集上泛化能力更强。
*   **计算效率:** 计算成本（GFLOPs）为 527.40，远低于 TEA 的 4335.82，减少约 7 倍计算量，内存使用也更低，适合实时应用。
*   **鲁棒性:** 在非独立同分布（non-i.i.d.）场景下保持最高平均准确率（60.99%），且对回放缓冲区大小和内容不敏感，即使缓冲区仅为源数据的 1%，性能几乎无下降。
*   **实验设置:** 实验覆盖多种损坏类型和严重程度，设置全面合理，充分验证了方法的提升效果。

## Further Thoughts

E PO TTA 将偏好优化与能量模型结合的思路非常启发性，这种方法可以通过偏好对直接优化分布，而无需显式建模残差或归一化常数，可推广至其他无监督或半监督学习任务中解决分布偏移问题；此外，其不确定性感知的梯度加权机制为动态调整样本影响提供了新思路，或许可以探索将其应用于大型语言模型（LLM）的测试时适应，解决自然语言处理中的分布偏移挑战。