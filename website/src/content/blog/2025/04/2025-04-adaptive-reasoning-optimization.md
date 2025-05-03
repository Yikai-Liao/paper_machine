---
title: "AdaR1: From Long-CoT to Hybrid-CoT via Bi-Level Adaptive Reasoning Optimization"
pubDatetime: 2025-04-30T14:01:45+00:00
slug: "2025-04-adaptive-reasoning-optimization"
type: "arxiv"
id: "2504.21659"
score: 0.737554648750315
author: "grok-3-latest"
authors: ["Haotian Luo", "Haiying He", "Yibo Wang", "Jinluan Yang", "Rui Liu", "Naiqiang Tan", "Xiaochun Cao", "Dacheng Tao", "Li Shen"]
tags: ["LLM", "Reasoning", "Efficiency Optimization", "Model Merging", "Preference Training"]
institution: ["Sun Yat-sen University", "China Agricultural University", "Tsinghua University", "Zhejiang University", "Didichuxing Co. Ltd", "Nanyang Technological University"]
description: "本文提出 AdaR1 框架，通过模型融合和双层偏好训练实现自适应推理，显著降低大型语言模型推理成本（平均长度减少超 50%）同时保持高性能。"
---

> **Summary:** 本文提出 AdaR1 框架，通过模型融合和双层偏好训练实现自适应推理，显著降低大型语言模型推理成本（平均长度减少超 50%）同时保持高性能。 

> **Keywords:** LLM, Reasoning, Efficiency Optimization, Model Merging, Preference Training

**Authors:** Haotian Luo, Haiying He, Yibo Wang, Jinluan Yang, Rui Liu, Naiqiang Tan, Xiaochun Cao, Dacheng Tao, Li Shen

**Institution(s):** Sun Yat-sen University, China Agricultural University, Tsinghua University, Zhejiang University, Didichuxing Co. Ltd, Nanyang Technological University


## Problem Background

大型语言模型（LLMs）在复杂推理任务中采用长链式思维（Long-CoT）显著提升了性能，但带来了高计算成本、延迟和资源消耗的效率瓶颈。
作者通过实证分析发现，Long-CoT 的收益高度依赖问题复杂性：复杂问题需要详细推理，而简单问题使用 Long-CoT 可能浪费资源甚至降低准确性。
因此，关键问题是设计自适应推理策略，根据输入问题特性动态调整推理深度和风格，以平衡性能和效率。

## Method

*   **核心思想:** 提出一个两阶段框架，通过构建混合推理模型并结合双层偏好训练，实现自适应推理，在性能和效率之间取得平衡。
*   **第一阶段 - 模型融合（Model Merging）:** 
    *   将 Long-CoT 模型和 Short-CoT 模型的参数通过线性融合（采用加权平均方式，融合系数 α 控制两种模型的贡献比例），构建一个混合推理模型（Hybrid Reasoning Model）。
    *   该模型能够生成长短两种推理风格，为后续自适应推理提供基础。
*   **第二阶段 - 双层偏好训练（Bi-Level Preference Training）:** 
    *   **组级偏好（Group-Level Preference）:** 通过比较 Long-CoT 和 Short-CoT 模型在给定问题上的准确性期望（基于采样多组响应并计算正确率），确定更适合当前问题的推理风格（长或短）。
    *   构建偏好对（preferred 和 rejected 响应对），用于指导模型选择合适的推理风格。
    *   **实例级偏好（Instance-Level Preference）:** 在选定的推理风格组内，进一步优化模型输出，偏好既正确又简洁的推理路径（选择最短的正确响应作为 preferred，较长的响应作为 rejected）。
    *   使用直接偏好优化（Direct Preference Optimization, DPO）方法，通过偏好数据集（包含组级和实例级偏好对）对混合模型进行微调，调整其生成行为。
*   **关键点:** 该方法不依赖于单一推理风格，而是通过自适应选择和优化，动态分配计算资源，避免不必要的推理开销，同时保持高准确性。

## Experiment

*   **有效性:** 在多个数学数据集（如 GSM8K, MATH）上，AdaR1 方法显著降低了推理长度（7B 模型平均减少 50.93%，1.5B 模型减少 43.28%），例如在 MATH 上减少 58%，GSM8K 上减少 74%，同时准确性下降极小（7B 模型下降 1.65%，1.5B 模型下降 1.21%），部分数据集甚至有所提升。
*   **优越性:** 相比其他方法如 CoT-Valve（长度减少 73.06% 但准确性下降 18.41%）和 O1-Pruner（准确性保持较好但长度减少效果较弱，仅 34.53%），AdaR1 在效率提升和性能保持之间取得了更好的平衡。
*   **实验设置合理性:** 实验覆盖了不同模型规模（1.5B 和 7B 参数）、多个数据集（GSM8K, MATH, AIME 等分布内测试集，以及 Olympiad, Minerva 等分布外测试集），并通过消融研究验证了组级和实例级偏好训练的贡献。数据集选择涵盖了不同难度问题，评估指标包括准确性和推理长度，设置全面合理。
*   **额外观察:** 图表分析显示，AdaR1 能根据问题难度自适应调整推理风格比例（高难度问题增加 Long-CoT 比例），在高难度任务上保持接近 Long-CoT 模型的准确性。

## Further Thoughts

AdaR1 的自适应推理策略展示了根据问题特性动态调整推理深度的潜力，这是否可以扩展到其他任务（如自然语言理解或多模态推理），通过更复杂的特征（如语义复杂度、上下文依赖性）指导自适应策略？
此外，模型融合与偏好训练的结合是否可以应用于其他多目标优化场景，例如在安全性和创造性之间平衡？
最后，是否可以在推理时引入更多动态调整机制（如基于实时反馈调整推理深度），进一步提升效率？