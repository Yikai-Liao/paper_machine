---
title: "AlignGuard-LoRA: Alignment-Preserving Fine-Tuning via Fisher-Guided Decomposition and Riemannian-Geodesic Collision Regularization"
pubDatetime: 2025-08-04T05:45:24+00:00
slug: "2025-08-alignment-preserving-finetuning"
type: "arxiv"
id: "2508.02079"
score: 0.5198053284464453
author: "grok-3-latest"
authors: ["Amitava Das", "Abhilekh Borah", "Vinija Jain", "Aman Chadha"]
tags: ["LLM", "Fine-Tuning", "Alignment", "Regularization", "Parameter Decomposition"]
institution: ["BITS Goa, India", "Manipal University, India", "Meta AI, USA", "Amazon AI, USA"]
description: "本文提出 AlignGuard-LoRA 框架，通过 Fisher 引导的参数分解和黎曼-测地正则化，在微调大型语言模型时降低高达 50% 的对齐漂移，同时保持任务性能。"
---

> **Summary:** 本文提出 AlignGuard-LoRA 框架，通过 Fisher 引导的参数分解和黎曼-测地正则化，在微调大型语言模型时降低高达 50% 的对齐漂移，同时保持任务性能。 

> **Keywords:** LLM, Fine-Tuning, Alignment, Regularization, Parameter Decomposition

**Authors:** Amitava Das, Abhilekh Borah, Vinija Jain, Aman Chadha

**Institution(s):** BITS Goa, India, Manipal University, India, Meta AI, USA, Amazon AI, USA


## Problem Background

大型语言模型（LLM）在微调过程中常出现对齐漂移（alignment drift），即安全性和行为约束（如拒绝有害回答）被削弱，即使是良性的任务导向微调也可能导致潜在安全风险。
关键问题是如何在微调时保留模型的对齐特性，同时不牺牲任务性能，尤其在多任务或持续学习场景下。

## Method

*   **核心思想:** 提出 AlignGuard-LoRA 框架，基于低秩适应（LoRA），通过将参数更新分解为对齐关键部分（ΔW_A）和任务特定部分（ΔW_T），并分别施加正则化，保护安全相关参数，同时支持任务学习。
*   **具体实现:**
    *   **Fisher 信息矩阵（FIM）分解:** 使用 FIM 计算任务损失的二阶敏感性，识别对齐关键方向，提取对齐相关子空间，并对 ΔW_A 施加基于 FIM 的正则化，限制其在敏感方向上的更新幅度，保护安全行为。
    *   **任务特定正则化:** 对 ΔW_T 施加基于信任区域或 Hessian 矩阵的正则化，确保任务学习的稳定性，避免过拟合或不稳定更新对整体模型的影响。
    *   **碰撞感知正则化:** 引入黎曼重叠（Riemannian overlap）和测地分离（geodesic separation）约束，避免 ΔW_A 和 ΔW_T 在参数空间中的干扰，确保两者更新方向和坐标不重叠，防止安全和任务目标之间的冲突。
*   **关键优势:** 不需要对齐监督数据，仅通过几何和信息理论工具实现对齐保护，同时保持任务适应性，适用于多种微调场景。

## Experiment

*   **任务性能:** 在 GLUE、SuperGLUE 和 HELM 基准测试中，AlignGuard-LoRA 的性能与标准 LoRA 和全模型微调相当，平均得分损失小于 0.4 个百分点，表明对齐保护未显著损害下游任务效果。
*   **对齐保留:** 在自制 DriftCheck 基准（10,000 个安全和不安全提示）上，将对齐漂移降低高达 50%，拒绝准确率从标准 LoRA 的 71.4% 提升至 92.3%，接近原始对齐模型（91.3%），毒性概率也显著降低。
*   **灾难性遗忘:** 通过 scaling law 分析，AlignGuard-LoRA 降低了遗忘幅度和残余漂移，表现出更好的预训练知识保留能力。
*   **实验设置合理性:** 实验覆盖多任务和数据集，评估了任务性能、对齐保留和遗忘行为，设置全面；DriftCheck 作为诊断工具针对对齐漂移设计，弥补了现有安全数据集不足，但主要基于 LLaMA 3 (7B)，对其他架构的泛化性需进一步验证。

## Further Thoughts

AlignGuard-LoRA 的结构化分解和几何正则化思路启发了我，是否可以将这种方法扩展到非语言模型（如视觉模型）中，保护特定能力不被后续任务破坏？此外，DriftCheck 的动态评估理念可以用于设计其他领域的诊断工具，捕捉模型在公平性或鲁棒性上的退化；同时，自适应正则化强度的设计可能根据任务和对齐需求动态调整，进一步提升框架的灵活性。