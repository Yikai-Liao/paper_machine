---
title: "Mixture of Reasonings: Teach Large Language Models to Reason with Adaptive Strategies"
pubDatetime: 2025-07-01T09:39:04+00:00
slug: "2025-07-mixture-reasoning-adaptive"
type: "arxiv"
id: "2507.00606"
score: 0.8710110457962957
author: "grok-3-latest"
authors: ["Tao Xiong", "Xavier Hu", "Wenyan Fan", "Shengyu Zhang"]
tags: ["LLM", "Reasoning", "Supervised Fine-Tuning", "Prompt Engineering", "Adaptive Strategies"]
institution: ["Dalian University of Technology", "Zhejiang University"]
description: "本文提出 Mixture of Reasoning (MoR) 框架，通过将多样化推理策略嵌入大型语言模型参数，实现无需任务特定提示的任务自适应推理，显著提升复杂任务性能。"
---

> **Summary:** 本文提出 Mixture of Reasoning (MoR) 框架，通过将多样化推理策略嵌入大型语言模型参数，实现无需任务特定提示的任务自适应推理，显著提升复杂任务性能。 

> **Keywords:** LLM, Reasoning, Supervised Fine-Tuning, Prompt Engineering, Adaptive Strategies

**Authors:** Tao Xiong, Xavier Hu, Wenyan Fan, Shengyu Zhang

**Institution(s):** Dalian University of Technology, Zhejiang University


## Problem Background

大型语言模型（LLMs）在复杂任务中依赖于手动设计的任务特定提示（如 Chain-of-Thought, CoT 和 Tree-of-Thought, ToT），这导致提示工程耗时且难以跨任务优化，限制了模型的适应性和效率。
论文旨在解决这一瓶颈，让模型能够自主选择和应用适合任务的推理策略，而无需外部提示设计。

## Method

*   **核心思想:** 提出 Mixture of Reasoning (MoR) 框架，通过将多样化的推理策略嵌入模型参数中，使 LLMs 能够自主、任务自适应地进行推理，摆脱对任务特定提示的依赖。
*   **具体实现:** MoR 框架分为两个阶段：
    *   **Thought Generation（思维生成）:** 利用闭源大模型（如 GPT-4o）生成大规模推理链模板（数量为 50、150、300、500 条），这些模板涵盖多步推理、类比推理和策略性思维等多种推理模式，为模型提供多样化的思维路径。
    *   **SFT Dataset Construction（监督微调数据集构建）:** 从多个基准数据集（如 HotpotQA, StrategyQA, MMLU 等）中抽取样本，为每个样本通过 GPT-4o 选择最适合的推理链模板，结合样本和模板生成提示，输入模型进行推理，筛选正确答案后构建监督微调（SFT）数据集，用于训练模型。
*   **关键点:** 通过监督微调将推理策略内化到模型参数中，使模型在无特定提示的情况下也能有效推理，同时减少人工干预。

## Experiment

*   **有效性:** 基于 Qwen2.5-7B-Instruct 基线模型，MoR_150 在 CoT 提示下准确率达到 0.730（提升 2.2%），在 IO 提示下达到 0.734（提升 13.5%），表现最佳，特别是在复杂任务（如 StrategyQA 和 MMLU）中优势明显。
*   **分析:** 增加推理链数量（如 MoR_500）并未持续提升性能，可能由于训练数据有限；扩展测试集（从 50 到 200 个样本）后，MoR_150 仍保持优势，表明方法的稳健性。
*   **实验设置:** 覆盖多个数据集（HotpotQA, StrategyQA, MMLU, BigTom, Trivial Creative Writing），任务类型多样，设置全面，但未详细讨论训练成本和计算开销。

## Further Thoughts

MoR 框架将推理策略内化到模型参数的思路启发我们思考是否可以通过自监督学习减少对闭源大模型的依赖；推理链数量与性能的关系提示未来可以探索动态调整推理链规模的方法；此外，是否可以将 MoR 与 RLHF 等训练范式结合，通过奖励机制优化推理策略的选择？