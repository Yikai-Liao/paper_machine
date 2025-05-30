---
title: "Budget-Adaptive Adapter Tuning in Orthogonal Subspaces for Continual Learning in LLMs"
pubDatetime: 2025-05-28T13:38:21+00:00
slug: "2025-05-budget-adaptive-continual-learning"
type: "arxiv"
id: "2505.22358"
score: 0.84104613629057
author: "grok-3-latest"
authors: ["Zhiyi Wan", "Wanrou Du", "Liang Li", "Miao Pan", "Xiaoqi Qin"]
tags: ["LLM", "Continual Learning", "Parameter Efficiency", "Orthogonal Subspace", "Fine-Tuning"]
institution: ["Beijing University of Posts and Telecommunications", "Pengcheng Laboratory", "University of Houston"]
description: "OA-Adapter 提出了一种参数高效的持续学习方法，通过单一端到端训练整合动态预算自适应和正交子空间学习，显著提升大型语言模型在持续学习中的准确性和参数效率。"
---

> **Summary:** OA-Adapter 提出了一种参数高效的持续学习方法，通过单一端到端训练整合动态预算自适应和正交子空间学习，显著提升大型语言模型在持续学习中的准确性和参数效率。 

> **Keywords:** LLM, Continual Learning, Parameter Efficiency, Orthogonal Subspace, Fine-Tuning

**Authors:** Zhiyi Wan, Wanrou Du, Liang Li, Miao Pan, Xiaoqi Qin

**Institution(s):** Beijing University of Posts and Telecommunications, Pengcheng Laboratory, University of Houston


## Problem Background

大型语言模型（LLMs）在持续学习（Continual Learning, CL）场景中面临灾难性遗忘问题，即在学习新任务时对先前任务性能的显著下降；
现有参数高效微调方法（如 Adapter 和 LoRA）在多任务序列学习中易导致任务间干扰，而正交子空间学习方法虽能缓解干扰，但固定预算分配忽略了任务复杂性和层级差异，参数利用效率低；
此外，现有预算自适应方法多采用多阶段优化，存在目标不对齐和计算复杂性问题。

## Method

*   **核心思想:** OA-Adapter 是一种参数高效的持续学习方法，在单一端到端训练阶段中结合动态预算自适应和正交子空间学习，既根据任务和层级需求动态分配参数预算，又通过正交约束减少任务间干扰和灾难性遗忘。
*   **模块结构:** 基于标准 Adapter 的瓶颈架构，OA-Adapter 去除了偏置项以便应用正交约束，并引入可训练的对角掩码矩阵 Γ，用于动态调整瓶颈维度；其前向计算为 y = x + W2 · Γ · W1 · x，其中 W1 和 W2 分别为下投影和上投影矩阵。
*   **动态瓶颈维度调整:** 通过一个可训练的阈值 τ 和软阈值机制，控制掩码矩阵 Γ 的稀疏性，动态激活或去激活维度，实现参数预算的自适应分配；该机制是双向的，允许维度在训练中重新激活，避免单向削减的局限性；有效瓶颈维度由非零掩码项数量决定（r_eff = ||γ||_0）。
*   **正交参数子空间约束:** 定义任务特定参数子空间为上投影矩阵 W2 的列空间，并对当前任务与历史任务的子空间施加正交约束，确保更新方向互不重叠；通过正交正则化损失项（L_orth）融入训练目标，平衡任务性能和知识保留，公式为 L_total = L_task + λ_orth · Σ L_orth[s,t]。
*   **优势:** 避免了多阶段优化的复杂性和不对齐问题，同时兼顾参数效率和知识保护。

## Experiment

*   **性能表现:** 在标准持续学习基准（5 任务）上，OA-Adapter 平均准确率（76.0%）显著优于前 SOTA 方法 O-LoRA（75.3%），接近多任务学习（MTL）理论上限（80.0%）；在 15 任务大规模基准上，OA-Adapter 也优于 O-LoRA（69.2% vs 68.7%），但仍低于 PerTaskFT 和 MTL，显示大规模任务挑战性。
*   **参数效率:** OA-Adapter 使用比 O-LoRA 少 46.6% 至 58.5% 的参数，同时保持或提升性能，验证了动态预算分配的有效性，避免固定预算的资源浪费。
*   **灾难性遗忘缓解:** 正交子空间约束显著减少遗忘，例如无约束时任务 2 性能降至近 0%，有约束时最大损失仅 14%。
*   **预算异质性验证:** 实验显示不同任务和层级对预算需求差异明显，初始任务预算较少，后续任务因需保留历史知识而增加，证明动态分配必要性。
*   **设置合理性:** 实验涵盖多种任务顺序、模型规模（T5-base, T5-large, T5-XL）和基准数据集，与多种基线方法对比，结果鲁棒；动态阈值策略优于固定阈值，表明其稳定性。

## Further Thoughts

动态预算分配与任务特性结合的思路值得扩展，未来可引入任务难度的显式评估（如数据分布差异或损失变化）以优化分配；
正交约束可探索非严格正交的软约束（如部分子空间重叠），在知识保留和新任务学习间寻找更灵活平衡；
论文观察到任务性能在后续训练初期有短暂恢复现象，类似人类记忆再激活，未来可研究是否通过记忆回放或子空间重激活机制挖掘模型中的潜在知识表示。