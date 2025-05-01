---
title: "AdaR1: From Long-CoT to Hybrid-CoT via Bi-Level Adaptive Reasoning Optimization"
pubDatetime: 2025-05-01T15:50:10Z
slug: "2025-04-adaptive-reasoning-optimization"
type: "arxiv"
id: "2504.21659"
score: 0.8396665148562898
author: "grok-3-latest"
authors: ["Haotian Luo", "Haiying He", "Yibo Wang", "Jinluan Yang", "Rui Liu", "Naiqiang Tan", "Xiaochun Cao", "Dacheng Tao", "Li Shen"]
tags: ["LLM", "Reasoning", "Sampling", "Pre-Training", "Post-Training"]
institution: ["Sun Yat-sen University", "China Agricultural University", "Tsinghua University", "Zhejiang University", "Didichuxing Co. Ltd", "Nanyang Technological University"]
description: "本文提出 AdaR1 框架，通过模型融合和双层偏好训练实现自适应推理，显著降低大型语言模型的推理成本同时保持性能。"
---

> **Summary:** 本文提出 AdaR1 框架，通过模型融合和双层偏好训练实现自适应推理，显著降低大型语言模型的推理成本同时保持性能。 

> **Keywords:** LLM, Reasoning, Sampling, Pre-Training, Post-Training
> **Recommendation Score:** 0.8396665148562898

**Authors:** Haotian Luo, Haiying He, Yibo Wang, Jinluan Yang, Rui Liu, Naiqiang Tan, Xiaochun Cao, Dacheng Tao, Li Shen
**Institution(s):** Sun Yat-sen University, China Agricultural University, Tsinghua University, Zhejiang University, Didichuxing Co. Ltd, Nanyang Technological University

## Problem Background

大型语言模型（LLMs）在复杂推理任务中采用长链式思维（Long-CoT）显著提升了性能，但带来了高昂的推理开销，包括计算成本、延迟和资源消耗，尤其在资源受限或需要快速响应的场景中成为瓶颈。
作者通过实证分析发现，Long-CoT 的收益高度依赖于问题复杂性：复杂问题需要详细推理，而简单问题使用 Long-CoT 不仅浪费资源，甚至可能降低准确性。
因此，论文旨在解决如何根据输入问题的特性自适应调整推理深度和风格，以在性能和效率之间取得平衡的关键问题。

## Method

* **核心思想：** 提出一个两阶段框架，通过构建混合推理模型并进行双层偏好训练，使模型能够根据问题复杂性自适应选择推理风格（Long-CoT 或 Short-CoT），并在选定风格内优化推理路径的简洁性和正确性。
* **第一阶段 - 模型融合（Model Merging）：** 通过线性参数插值（θ_H = αθ_L + (1-α)θ_S，其中 α 为平衡系数），将 Long-CoT 模型和 Short-CoT 模型融合为一个混合推理模型（Hybrid Reasoning Model），使其具备生成长短两种推理风格的能力，为自适应推理奠定基础。
* **第二阶段 - 双层偏好训练（Bi-Level Preference Training）：**
  - **组级偏好（Group-Level Preference）：** 针对输入问题，采样 Long-CoT 和 Short-CoT 模型的响应，计算每组的准确率期望（通过指标函数），并基于偏好阈值（ϵ）确定更适合的推理风格；随后使用直接偏好优化（DPO）训练模型，使其学习根据问题复杂性选择合适的推理风格。
  - **实例级偏好（Instance-Level Preference）：** 在选定的风格组内，进一步优化推理路径；通过比较同一组内的响应，选择最短的正确响应作为优选样本，选择最长的响应作为劣选样本，用于 DPO 训练，鼓励模型生成既正确又简洁的推理。
* **关键点：** 该方法不修改原始模型结构，仅通过融合和训练调整推理行为；双层训练策略在全局（风格选择）和局部（路径优化）两个层面动态分配计算资源，从而提升效率。

## Experiment

* **有效性：** 在多个数学数据集（GSM8K, MATH, AIME 等）上，AdaR1 方法显著降低了推理长度（7B 模型平均减少 50.93%，1.5B 模型减少 43.28%），相比其他基线方法（如 O1-Pruner 和 DPO）表现出更优的效率提升。
* **性能保持：** 准确率仅略有下降（7B 模型下降 1.65%，1.5B 模型下降 1.21%），远低于其他广域优化方法（如 CoT-Valve 和 Naive Merge，下降超过 10%），表明 AdaR1 在效率和性能之间取得了更好的权衡。
* **实验设置合理性：** 实验覆盖了不同难度的数学问题（通过 MATH 数据集的难度分级），并使用了分布内（GSM8K, MATH, AIME）和分布外（Olympiad, Minerva）测试集，确保了结果的泛化性；消融研究验证了双层训练的贡献，组级和实例级偏好训练结合效果最佳。
* **潜在局限：** 方法对模型规模的依赖性（7B 和 1.5B 模型效果有差异）以及对特定领域（数学推理）的适用性需进一步探索；采样次数和偏好阈值的选择可能影响结果，论文未详细讨论其鲁棒性。

## Further Thoughts

AdaR1 的自适应推理思想可以扩展到其他领域（如对话系统或代码生成），是否可以设计一个通用的复杂性评估模块，实时判断任务需求并分配计算资源？
此外，是否可以通过融合更多推理风格（如分层推理或并行推理）实现更灵活的自适应推理？
另外，AdaR1 在推理时需要额外采样和偏好计算，是否可以通过预训练一个复杂性预测器减少额外开销，进一步优化效率？