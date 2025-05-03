---
title: "DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition"
pubDatetime: 2025-04-30T16:57:48+00:00
slug: "2025-04-deepseek-prover-v2"
type: "arxiv"
id: "2504.21801"
score: 0.5729620052484906
author: "grok-3-latest"
authors: ["Z.Z. Ren", "Zhihong Shao", "Junxiao Song", "Huajian Xin", "Haocheng Wang", "Wanjia Zhao", "Liyue Zhang", "Zhe Fu", "Qihao Zhu", "Dejian Yang", "Z.F. Wu", "Zhibin Gou", "Shirong Ma", "Hongxuan Tang", "Yuxuan Liu", "Wenjun Gao", "Daya Guo", "Chong Ruan"]
tags: ["LLM", "Formal Proof", "Subgoal Decomposition", "Reinforcement Learning", "Reasoning"]
institution: ["DeepSeek-AI"]
description: "本文提出了一种基于子目标分解和强化学习的训练框架，显著提升了大型语言模型在形式化定理证明中的性能，并在多个基准数据集上取得了最先进的成果。"
---

> **Summary:** 本文提出了一种基于子目标分解和强化学习的训练框架，显著提升了大型语言模型在形式化定理证明中的性能，并在多个基准数据集上取得了最先进的成果。 

> **Keywords:** LLM, Formal Proof, Subgoal Decomposition, Reinforcement Learning, Reasoning

**Authors:** Z.Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z.F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, Chong Ruan

**Institution(s):** DeepSeek-AI


## Problem Background

大型语言模型（LLMs）在自然语言推理中表现出色，但由于缺乏形式化系统的严谨性，难以直接应用于形式化定理证明（如 Lean 4）。
论文旨在弥合自然语言推理与形式化证明之间的差距，解决如何利用 LLMs 的推理能力高效生成可验证的形式化证明这一关键问题。

## Method

*   **核心思想:** 通过子目标分解（Subgoal Decomposition）将复杂定理证明问题拆分为一系列较小的子目标，并结合强化学习优化模型的推理和证明能力。
*   **具体实现:**
    *   **递归定理证明管道:** 利用 DeepSeek-V3 模型生成自然语言证明草稿并将其形式化为 Lean 4 代码（包含 sorry 占位符），分解为子目标；随后使用参数较小的 7B 模型解决子目标，降低计算成本。
    *   **冷启动数据合成:** 将 DeepSeek-V3 的链式推理（Chain-of-Thought, CoT）与形式化证明步骤结合，生成包含自然语言推理和形式化代码的高质量训练数据。
    *   **课程学习:** 基于子目标分解设计逐步增加难度的训练任务，帮助模型逐步适应复杂问题。
    *   **强化学习优化:** 采用 Group Relative Policy Optimization (GRPO) 算法，通过二元奖励（正确/错误）优化模型性能，并引入一致性奖励确保生成的证明与子目标分解结构对齐。
    *   **双阶段训练:** 训练两种模式——高效的非链式推理模式（non-CoT）用于快速生成证明，高精度的链式推理模式（CoT）用于复杂推理场景。
*   **关键创新:** 将自然语言推理与形式化证明结合，通过子目标分解降低问题复杂度，并利用强化学习提升模型在形式化定理证明中的表现。

## Experiment

*   **有效性:** DeepSeek-Prover-V2-671B 在多个基准数据集上取得显著成果，如 MiniF2F-test 上 Pass@8192 准确率达 88.9%，ProofNet-test 上 Pass@1024 达 37.1%，PutnamBench 解决 49/658 个问题，ProverBench-AIME 解决 6/15 个问题，均优于其他最先进模型。
*   **优越性:** CoT 模式相较 non-CoT 模式有显著提升，表明显式推理步骤对形式化证明至关重要；随着样本预算增加，模型性能持续提升，显示出良好的采样效率。
*   **实验设置合理性:** 实验覆盖高中竞赛到大学水平的多种数学问题，验证了模型在不同难度和领域上的泛化能力；同时对比了不同模型规模（7B vs 671B）和推理模式（CoT vs non-CoT），设置较为全面。
*   **不足与开销:** CoT 模式下生成 token 数量显著增加（671B 模型平均 6751.9 个 token），计算成本较高，可能限制实际应用场景。

## Further Thoughts

子目标分解与强化学习的结合为解决复杂问题提供了新思路，可推广至代码生成或逻辑推理等领域；自然语言与形式化推理的统一提示我们可以在其他任务中尝试将非结构化知识与结构化系统结合；课程学习的应用表明逐步增加任务难度可能对训练任务导向的 AI 系统具有普遍意义。