---
title: "Task-Core Memory Management and Consolidation for Long-term Continual Learning"
pubDatetime: 2025-05-15T04:22:35+00:00
slug: "2025-05-long-cl-memory"
type: "arxiv"
id: "2505.09952"
score: 0.6713065912624082
author: "grok-3-latest"
authors: ["Tianyu Huai", "Jie Zhou", "Yuxuan Cai", "Qin Chen", "Wen Wu", "Xingjiao Wu", "Xipeng Qiu", "Liang He"]
tags: ["LLM", "Continual Learning", "Memory Management", "Catastrophic Forgetting", "Sample Selection"]
institution: ["East China Normal University", "Nanyang Technological University", "Fudan University"]
description: "本文提出 `Long-CL` 框架，通过任务核心记忆管理和长期记忆巩固机制显著缓解长期持续学习中的灾难性遗忘，并在自建的多模态和文本基准数据集上实现了最先进的性能。"
---

> **Summary:** 本文提出 `Long-CL` 框架，通过任务核心记忆管理和长期记忆巩固机制显著缓解长期持续学习中的灾难性遗忘，并在自建的多模态和文本基准数据集上实现了最先进的性能。 

> **Keywords:** LLM, Continual Learning, Memory Management, Catastrophic Forgetting, Sample Selection

**Authors:** Tianyu Huai, Jie Zhou, Yuxuan Cai, Qin Chen, Wen Wu, Xingjiao Wu, Xipeng Qiu, Liang He

**Institution(s):** East China Normal University, Nanyang Technological University, Fudan University


## Problem Background

本文聚焦于长期持续学习（Long-term Continual Learning, Long-CL），即模型需要在长时间内从大量任务流中顺序学习新知识，同时保留旧知识，类似于人类学习方式。
传统持续学习（CL）方法在任务数量较少时有效，但在长期CL场景下，由于任务数量大幅增加，灾难性遗忘问题（Catastrophic Forgetting）变得更加严重，现有方法难以维持性能。
此外，缺乏针对长期CL的全面基准数据集，限制了相关研究的评估和比较。

## Method

*   **核心思想:** 提出一个受人类记忆机制启发的框架 `Long-CL`，通过任务核心记忆管理（MemMan）和长期记忆巩固（MemCon）两大组件，在长期任务序列中动态平衡新旧知识，缓解灾难性遗忘。
*   **任务核心记忆管理（MemMan）:** 
    *   **任务核心记忆索引（Task-Core Memory Indexing）:** 通过计算当前任务模型与之前模型参数的差异，识别对当前任务贡献最大的关键记忆单元（Top-K 单位），并将其索引存储在掩码矩阵中，用于后续更新。
    *   **自适应记忆更新（Adaptive Memory Updating）:** 基于任务原型之间的语义相似性，计算自适应权重（α），以动态调整新旧知识的融合比例；在早期任务中权重较大以鼓励新知识吸收，在后期任务中权重逐渐减小以保护旧知识；同时结合掩码矩阵进行位置感知的记忆融合，确保关键记忆单元得到更多关注。
*   **长期记忆巩固（MemCon）:** 
    *   **硬样本选择（Hard Sample Selection）:** 针对当前任务，计算样本与任务原型的语义距离，选择距离最远的样本（通常是困难样本或边界案例），以增强模型对挑战性数据的处理能力。
    *   **差异样本选择（Differential Sample Selection）:** 针对历史任务，选择与所有历史任务原型累积距离最小的样本，确保跨任务的语义一致性，同时设置最小距离阈值以避免过于偏向单一任务。
    *   **经验回放优化:** 将硬样本和差异样本结合，构建回放缓冲区，用于在学习新任务时回放关键样本，强化长期知识保留和泛化能力。
*   **技术细节:** 采用低秩适应（LoRA）技术进行参数高效微调，仅更新Transformer块中的前馈神经网络层参数，保持原始模型参数冻结，降低计算成本。

## Experiment

*   **有效性:** 在自建的两个基准数据集 `MMLongCL-Bench`（多模态，21个任务）和 `TextLongCL-Bench`（文本，30个任务）上，`Long-CL` 的最终平均性能（AP）分别比最先进基线提高了7.4% 和 6.5%，平均遗忘（AF）指标显著优于基线，达到负值（-9.93% 和 -0.89%），表明模型在旧任务上的性能甚至超过初始训练结果，展现出强大的反向迁移能力。
*   **对比分析:** 相较于多种基线方法（如 EWC, ER, O-LoRA, CL-MoE），`Long-CL` 在所有任务类别上均有显著提升，尤其在视觉-语言任务（遗忘问题更严重）中表现突出。
*   **实验设置合理性:** 两个基准涵盖多模态和文本任务，任务类型和数据分布多样，实验设计全面；同时对任务顺序、缓冲区大小、关键记忆单元比例（K 值）等进行了敏感性分析，验证了方法的鲁棒性。
*   **消融研究:** MemMan 和 MemCon 组件均对性能有贡献，MemCon 的回放机制在缓解遗忘方面作用更大，二者结合效果最佳。
*   **资源开销:** 方法在缓冲区大小为20%时即可接近多任务学习的性能，实现了性能与资源效率的良好平衡。

## Further Thoughts

受人类记忆机制启发的记忆管理和巩固策略为长期学习提供了新思路，是否可以进一步借鉴遗忘曲线等认知机制优化更新频率？
自适应权重基于任务相似性的设计启发我们是否可以通过图结构建模任务关系以更精细地分配权重？
样本选择策略的双重机制（硬样本与差异样本）是否可以结合强化学习动态调整选择标准？
新基准数据集的构建为长期CL研究奠定了基础，是否可以扩展到更多模态（如音频、视频）或跨领域任务以模拟更真实场景？