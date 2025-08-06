---
title: "Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models"
pubDatetime: 2025-08-03T20:07:15+00:00
slug: "2025-08-continual-pretraining-replay"
type: "arxiv"
id: "2508.01908"
score: 0.905791594609492
author: "grok-3-latest"
authors: ["Istabrak Abbes", "Gopeshh Subbaraj", "Matthew Riemer", "Nizar Islah", "Benjamin Thérien", "Tsuguchika Tabaru", "Hiroaki Kingetsu", "Sarath Chandar", "Irina Rish"]
tags: ["LLM", "Continual Learning", "Experience Replay", "Gradient Alignment", "Pre-Training"]
institution: ["Université de Montréal", "Mila – Quebec AI Institute", "Chandar Research Lab", "IBM Research", "Fujitsu Research", "Polytechnique Montréal"]
description: "本文提出了一种结合经验回放和梯度对齐的元经验回放方法，用于大型语言模型的持续预训练，显著缓解灾难性遗忘并提升新任务适应性和下游任务泛化能力，同时保持高效计算。"
---

> **Summary:** 本文提出了一种结合经验回放和梯度对齐的元经验回放方法，用于大型语言模型的持续预训练，显著缓解灾难性遗忘并提升新任务适应性和下游任务泛化能力，同时保持高效计算。 

> **Keywords:** LLM, Continual Learning, Experience Replay, Gradient Alignment, Pre-Training

**Authors:** Istabrak Abbes, Gopeshh Subbaraj, Matthew Riemer, Nizar Islah, Benjamin Thérien, Tsuguchika Tabaru, Hiroaki Kingetsu, Sarath Chandar, Irina Rish

**Institution(s):** Université de Montréal, Mila – Quebec AI Institute, Chandar Research Lab, IBM Research, Fujitsu Research, Polytechnique Montréal


## Problem Background

大型语言模型（LLMs）在面对新数据时通常需要从头重新训练，这导致了巨大的计算资源浪费和环境成本。
持续预训练（Continual Pre-Training, CPT）旨在让模型在已有预训练基础上逐步适应新数据分布，但新数据的引入往往引发分布偏移，导致‘灾难性遗忘’（Catastrophic Forgetting），即模型在新任务上学习时丢失对旧任务的知识。
论文致力于解决稳定性-可塑性困境（Stability-Plasticity Dilemma），探索如何在更新模型时既能学习新知识，又能保留旧知识。

## Method

*   **核心思想:** 重新审视持续学习中的经验回放（Experience Replay）和梯度对齐（Gradient Alignment）方法，提出一种高效的元经验回放（Meta-Experience Replay, MER）框架，用于 LLMs 的持续预训练，以缓解灾难性遗忘并提升新任务适应性。
*   **经验回放实现:** 维护一个存储过去数据的缓冲区，在训练新数据时按一定比例（如 25% 或 50%）混合旧数据样本，以稳定学习过程；通过磁盘存储和异步预取机制实现高效的‘无限’回放缓冲区，避免内存限制，同时优化计算效率。
*   **梯度对齐实现:** 采用 Reptile 算法（一种元学习方法），通过定期对模型参数进行插值更新（每 k 个批次更新一次），促进梯度间的正向转移（Transfer）并减少干扰（Interference），以保护旧任务知识。
*   **元经验回放（MER）:** 结合经验回放和梯度对齐，利用 Reptile 的元优化机制增强回放效果，同时保持计算和内存开销极低（Reptile 更新仅每 500 个批次增加微小 FLOPs）。
*   **关键创新:** 首次将梯度对齐技术应用于 LLMs 的持续预训练场景，并通过高效实现（如磁盘存储）适应大规模数据需求，同时探索回放比例与计算资源的权衡。

## Experiment

*   **有效性:** 实验基于 Llama 架构的 Spectra 模型（规模从 99M 到 6B 参数），在多语言数据（英语、法语、德语、阿拉伯语、日语，每种 1000 亿 token）上进行持续预训练；经验回放显著降低遗忘分数，尤其在 50% 回放比例下效果最佳；结合 Reptile 的 MER 方法进一步减少遗忘，甚至实现负遗忘（即回溯性转移）。
*   **性能提升:** 在下游任务（如 HellaSwag、PiQA、PubMedQA）上，MER 方法（特别是 25% 回放+Reptile）在中小规模模型（560M）上表现出显著泛化能力提升，甚至超越联合训练基线；6B 模型在 50% 回放+Reptile 下达到最佳下游性能（平均准确率 77.1）。
*   **计算效率:** 25% 回放比例比增加模型规模更具计算效率（FLOPs 对比），但 50% 回放比例的额外收益不如扩大模型规模；Reptile 的加入几乎不增加计算开销，却显著提升稳定性和可塑性。
*   **实验合理性:** 实验设置全面，涵盖多种模型规模和任务数量（3 任务和 5 任务序列），数据量巨大，评价指标（如遗忘分数、保留损失、学习损失、下游任务性能）多维度；但任务序列固定且数量有限，未完全反映真实世界复杂场景。

## Further Thoughts

论文揭示了经验回放与计算资源的权衡关系，启发我们思考是否可以通过动态调整回放比例（例如根据任务难度或分布差异）进一步优化效率；梯度对齐（Reptile）在低开销下显著提升性能，提示元学习方法在 LLMs 持续学习中的潜力，是否可以结合其他元学习算法（如 MAML）进一步优化；磁盘存储回放缓冲区的设计为大规模训练提供了解决方案，是否可以推广到强化学习或多模态学习领域；多语言持续学习的分布差异挑战，是否可以根据语言间的语义或语法相似性设计针对性回放策略？