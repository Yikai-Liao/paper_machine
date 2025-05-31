---
title: "Budget-Adaptive Adapter Tuning in Orthogonal Subspaces for Continual Learning in LLMs"
pubDatetime: 2025-05-28T13:38:21+00:00
slug: "2025-05-budget-adaptive-adapter"
type: "arxiv"
id: "2505.22358"
score: 0.84104613629057
author: "grok-3-latest"
authors: ["Zhiyi Wan", "Wanrou Du", "Liang Li", "Miao Pan", "Xiaoqi Qin"]
tags: ["LLM", "Continual Learning", "Parameter Efficiency", "Orthogonal Subspace", "Budget Adaptation"]
institution: ["Beijing University of Posts and Telecommunications", "Pengcheng Laboratory", "University of Houston"]
description: "本文提出 OA-Adapter，通过动态预算分配和正交子空间约束，在持续学习中显著提升大型语言模型性能和参数效率，为参数高效微调和知识保留提供新范式。"
---

> **Summary:** 本文提出 OA-Adapter，通过动态预算分配和正交子空间约束，在持续学习中显著提升大型语言模型性能和参数效率，为参数高效微调和知识保留提供新范式。 

> **Keywords:** LLM, Continual Learning, Parameter Efficiency, Orthogonal Subspace, Budget Adaptation

**Authors:** Zhiyi Wan, Wanrou Du, Liang Li, Miao Pan, Xiaoqi Qin

**Institution(s):** Beijing University of Posts and Telecommunications, Pengcheng Laboratory, University of Houston


## Problem Background

大型语言模型（LLMs）在持续学习（Continual Learning, CL）场景中常遭受灾难性遗忘（Catastrophic Forgetting），即在新任务训练时先前任务性能严重下降。
现有正交子空间学习方法虽能通过限制任务更新到互不干扰的子空间来缓解任务间干扰，但通常采用固定预算分配，忽略任务复杂性和层级差异，导致参数利用效率低下；此外，现有预算自适应方法多采用多阶段优化设计，存在目标不对齐和计算复杂性问题，不适合持续学习场景。

## Method

*   **核心思想:** 提出 OA-Adapter（Orthogonal Adaptive Adapter），一种参数高效的持续学习框架，通过统一动态预算分配和正交子空间学习，在单阶段端到端训练中同时优化任务性能和参数效率，缓解灾难性遗忘。
*   **架构设计:** 基于标准 Adapter 模块，采用瓶颈结构，通过下投影（Down-Projection）和上投影（Up-Projection）层减少参数量，去除偏置项以便于正交约束实现，保持原始模型权重冻结，仅更新轻量级任务特定参数。
*   **动态预算分配机制:** 引入可训练的对角掩码矩阵（Diagonal Masking Matrix）和可学习阈值（Trainable Threshold），通过软阈值机制（Soft Thresholding）动态调整瓶颈维度（Bottleneck Dimension），实现维度双向激活或去激活，根据任务难度和层级需求自适应分配参数预算，避免固定预算的低效问题。
*   **正交子空间约束:** 对当前任务的参数子空间与历史任务的动态分配子空间施加正交约束，确保任务间更新方向互不干扰，通过正交正则化损失（Orthogonality Regularization Loss）量化子空间间内积并将其驱动至零，有效保留先前知识。
*   **端到端训练优势:** 将预算分配与任务优化统一在单阶段训练中，避免多阶段方法中优化目标与预算分配不对齐的问题，降低计算和工程开销，提升持续学习实用性。

## Experiment

*   **性能提升:** 在标准持续学习基准（5 个任务）上，OA-Adapter 平均准确率达到 76.0%，优于前最佳方法 O-LoRA（75.3%），接近多任务学习（MTL）理论上限（80.0%）；在更大规模基准（15 个任务）上，准确率达 69.2%，仍优于 O-LoRA（68.7%），尽管低于 PerTaskFT 和 MTL，显示大规模任务挑战性。
*   **参数效率:** 通过动态预算分配，OA-Adapter 使用比 O-LoRA 少 46.6% 至 58.5% 的参数，同时保持或提升性能，验证了其高效性。
*   **灾难性遗忘缓解:** 对比实验显示，正交子空间约束显著减少遗忘，例如无约束时任务性能降至近 0%，有约束时最严重下降仅 14%。
*   **实验设置合理性:** 实验覆盖多种任务顺序（Order 1-6）和模型规模（T5-base, T5-large, T5-XL），测试不同初始预算条件，设置全面；任务顺序多样性确保结果鲁棒性，模型规模扩展验证方法可扩展性；动态预算分配在任务和层级上的异质性与假设一致，初始任务稀疏性高，后续任务需更多参数。

## Further Thoughts

动态预算分配机制可扩展至其他参数高效微调任务，如单任务或多模态学习，通过自适应参数分配提升效率；正交约束与模型容量权衡可进一步优化，探索自适应正则化强度；任务无关训练和遗忘中潜在知识恢复现象为持续学习提供了新视角，未来可通过元学习或聚类方法实现任务无关训练，或利用知识再激活设计更高效策略。