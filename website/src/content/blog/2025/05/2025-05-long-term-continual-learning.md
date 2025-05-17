---
title: "Task-Core Memory Management and Consolidation for Long-term Continual Learning"
pubDatetime: 2025-05-15T04:22:35+00:00
slug: "2025-05-long-term-continual-learning"
type: "arxiv"
id: "2505.09952"
score: 0.6713065912624082
author: "grok-3-latest"
authors: ["Tianyu Huai", "Jie Zhou", "Yuxuan Cai", "Qin Chen", "Wen Wu", "Xingjiao Wu", "Xipeng Qiu", "Liang He"]
tags: ["Continual Learning", "Catastrophic Forgetting", "Memory Management", "Sample Selection", "Task Adaptation"]
institution: ["East China Normal University", "Nanyang Technological University", "Fudan University"]
description: "本文提出 `Long-CL` 框架，通过任务核心记忆管理和长期记忆巩固机制，显著缓解了长期持续学习中的灾难性遗忘问题，并在多模态和文本基准上取得了最优性能。"
---

> **Summary:** 本文提出 `Long-CL` 框架，通过任务核心记忆管理和长期记忆巩固机制，显著缓解了长期持续学习中的灾难性遗忘问题，并在多模态和文本基准上取得了最优性能。 

> **Keywords:** Continual Learning, Catastrophic Forgetting, Memory Management, Sample Selection, Task Adaptation

**Authors:** Tianyu Huai, Jie Zhou, Yuxuan Cai, Qin Chen, Wen Wu, Xingjiao Wu, Xipeng Qiu, Liang He

**Institution(s):** East China Normal University, Nanyang Technological University, Fudan University


## Problem Background

本文聚焦于长期持续学习（Long-term Continual Learning, Long-CL），即模型需要在长时间内从大量任务流中顺序学习新知识，同时避免灾难性遗忘（Catastrophic Forgetting），即新任务学习导致旧任务性能下降的问题。
与传统持续学习（CL）相比，长期CL涉及更多任务和更长的学习周期，遗忘问题更为严重，而现有方法在任务数量大幅增加时表现不佳，难以平衡新旧知识的保留与适应。

## Method

* **框架概述**：提出受人类记忆机制启发的 `Long-CL` 框架，包含两个核心模块：任务核心记忆管理（Task-Core Memory Management, MemMan）和长期记忆巩固（Long-term Memory Consolidation, MemCon），旨在通过参数级和样本级的策略缓解遗忘问题。
* **MemMan 模块**：
  * **任务核心记忆索引（Task-Core Memory Indexing）**：通过计算当前任务模型与前一任务模型参数的差异，识别对当前任务贡献最大的记忆单元（Top-K 单位），并将其存储在掩码矩阵中，用于指导后续更新。
  * **自适应记忆更新（Adaptive Memory Updating）**：基于任务原型（Task Prototype）间的语义距离，计算自适应权重因子（Adaptive Weighting Factor），动态平衡新任务知识注入与旧任务知识保留的程度，确保早期任务更新更激进，后期更保守。
* **MemCon 模块**：
  * **硬样本选择（Hard Sample Selection）**：针对当前任务，基于样本与任务原型的欧几里得距离，筛选出最具挑战性的样本（即远离原型的样本），以增强模型对困难案例的学习。
  * **差异样本选择（Differential Sample Selection）**：筛选与历史任务原型全局一致的样本，同时避免过于偏向单一任务，确保跨任务的泛化性，构建回放缓冲区（Replay Buffer）用于经验回放。
* **技术细节**：采用低秩适应（LoRA）技术进行参数高效微调，仅更新Transformer块中的前馈网络层参数，保持原始模型冻结，降低计算成本。

## Experiment

* **数据集与指标**：在自建的两个基准数据集 `MMLongCL-Bench`（多模态，21个任务）和 `TextLongCL-Bench`（文本，30个任务）上进行实验，使用最终平均性能（AP）和平均遗忘率（AF）作为评价指标。
* **性能提升**：`Long-CL` 在两个基准上的 AP 分别达到 51.93% 和 60.12%，比最强基线提升 7.4% 和 6.5%；AF 为负值（-9.93% 和 -0.89%），表明存在反向迁移能力，即旧任务性能优于初始训练结果。
* **对比分析**：与多种持续学习方法（正则化、架构、回放等）相比，`Long-CL` 在任务多样性和长期学习场景下表现更优，尤其在视觉-语言任务上遗忘缓解更显著，但仍未完全达到多任务学习的上限。
* **实验全面性**：实验分析了任务顺序、缓冲区大小、超参数（如自适应权重和记忆单元比例）的影响，设置合理且覆盖多种场景；同时指出视觉-语言任务因数据分布离散性更大，遗忘问题较文本任务更严重。
* **成本与局限**：方法增加了样本选择和记忆管理的计算开销，但通过 LoRA 微调降低了整体训练成本；性能虽接近多任务学习，但仍有改进空间。

## Further Thoughts

论文中基于任务语义距离的自适应更新权重机制启发我们思考是否可以通过任务分组或聚类策略进一步优化记忆更新，例如将相似任务的记忆单元优先整合；此外，MemCon 的样本选择策略是否可以结合强化学习动态调整样本优先级，而不仅仅依赖距离度量；另外，是否可以引入分层记忆机制，将短期和长期记忆分开管理，模拟人类大脑的海马体与皮层分工，以更高效地处理长期持续学习中的知识保留与更新。