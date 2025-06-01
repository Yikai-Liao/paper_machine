---
title: "DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning"
pubDatetime: 2025-05-29T17:59:39+00:00
slug: "2025-05-deep-theorem-rl"
type: "arxiv"
id: "2505.23754"
score: 0.5574694243728724
author: "grok-3-latest"
authors: ["Ziyin Zhang", "Jiahao Xu", "Zhiwei He", "Tian Liang", "Qiuzhi Liu", "Yansi Li", "Linfeng Song", "Zhengwen Liang", "Zhuosheng Zhang", "Rui Wang", "Zhaopeng Tu", "Haitao Mi", "Dong Yu"]
tags: ["LLM", "Natural Language", "Theorem Proving", "Reinforcement Learning", "Reasoning"]
institution: ["Tencent", "Shanghai Jiao Tong University"]
description: "DeepTheorem 通过大规模自然语言定理数据集和 RL-Zero 强化学习方法，显著提升了大型语言模型在非形式化定理证明中的推理能力，超越传统监督微调框架。"
---

> **Summary:** DeepTheorem 通过大规模自然语言定理数据集和 RL-Zero 强化学习方法，显著提升了大型语言模型在非形式化定理证明中的推理能力，超越传统监督微调框架。 

> **Keywords:** LLM, Natural Language, Theorem Proving, Reinforcement Learning, Reasoning

**Authors:** Ziyin Zhang, Jiahao Xu, Zhiwei He, Tian Liang, Qiuzhi Liu, Yansi Li, Linfeng Song, Zhengwen Liang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu

**Institution(s):** Tencent, Shanghai Jiao Tong University


## Problem Background

定理证明是评估大型语言模型（LLM）复杂推理能力的重要领域，但传统自动定理证明（ATP）依赖形式化证明系统（如 Lean, Coq），与 LLM 基于自然语言的非形式化知识优势不匹配，限制了其潜力。
本文提出 DeepTheorem 框架，旨在通过自然语言驱动的非形式化定理证明，释放 LLM 的数学推理能力，解决形式化系统带来的障碍。

## Method

*   **数据集构建 (DeepTheorem Dataset):** 构建了一个包含 121K 个 IMO 级别非形式化数学定理和证明的大型基准数据集，覆盖多个数学领域（如代数、几何、数论），并通过严格的去污染流程避免与测试基准重叠。数据集标注了正确性、难度（1-9 级，聚焦高难度 6-9 级）和主题类别，同时生成了可验证的定理变体（通过逻辑蕴含或矛盾变换），为后续强化学习提供支持。
*   **训练策略 (RL-Zero):** 提出了一种专为非形式化定理证明设计的强化学习方法 RL-Zero，利用定理变体的二元奖励机制（证明或反驳），通过 GRPO 算法训练模型，鼓励模型探索并生成逻辑严谨的证明，而非单纯模仿监督微调（SFT）中的示例。系统提示中加入 <think> 标签以激励详细推理，并通过奖励函数和约束条件（如空白比例、重复字符检查）防止模型崩溃。
*   **评估框架:** 设计了两种评估方式：结果评估（Outcome Evaluation）通过定理变体的正确性判断模型证明能力；过程评估（Process Evaluation）从逻辑有效性、完整性、正确性和清晰度四个维度，借助 GPT-4o 作为评判模型，评估证明过程的质量。

## Experiment

*   **有效性:** 在多个基准数据集（FIMO, HMMT, Putnam）上，DeepTheorem 结合 RL-Zero 训练的 7B 模型显著优于监督微调（SFT）和 OpenR1-Math-Proof 数据集，尤其在过程评估中提升明显，例如在 Putnam 数据集上，DeepTheorem-RL-7B 的过程评分达到 42.20，远高于 SFT 的 33.50。
*   **参数效率:** 实验显示 DeepTheorem-RL 在 1.5B 到 7B 模型规模上均表现出色，参数效率高于 Qwen2.5 系列和部分商用模型（如 o1），表明方法在小规模模型上也能取得显著效果。
*   **对比分析:** 与商用模型（如 o3-mini, Gemini2.5-Pro）和开源模型相比，DeepTheorem-RL-7B 在同等规模下达到或超越了大多数模型的性能，特别是在 HMMT 和 Putnam 基准上，平均结果评分达到 47.22，过程评分达到 34.04。
*   **实验设置合理性:** 实验涵盖不同模型规模（1.5B, 3B, 7B）、训练策略（SFT vs RL）、多个测试基准，并通过数据去污染和难度标注确保公平性；但未详细讨论 RL-Zero 的计算成本和收敛速度，可能对实际应用有一定影响。

## Further Thoughts

DeepTheorem 的自然语言定理证明思路启发我们将 LLM 的预训练知识优势扩展到其他复杂推理领域（如法律推理或科学假设验证），而 RL-Zero 通过定理变体生成二元奖励的探索机制可能适用于开放性问题解决任务；此外，过程评估框架的多维度推理质量评估方法对构建更可解释的 AI 系统具有重要意义，或许能帮助解决 LLM 输出‘黑箱’问题。