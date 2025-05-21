---
title: "J4R: Learning to Judge with Equivalent Initial State Group Relative Preference Optimization"
pubDatetime: 2025-05-19T16:50:35+00:00
slug: "2025-05-eis-grpo-judge"
type: "arxiv"
id: "2505.13346"
score: 0.5606450727018389
author: "grok-3-latest"
authors: ["Austin Xu", "Yilun Zhou", "Xuan-Phi Nguyen", "Caiming Xiong", "Shafiq Joty"]
tags: ["LLM", "Reinforcement Learning", "Evaluation", "Reasoning", "Bias Mitigation"]
institution: ["Salesforce AI Research"]
description: "本文提出 EIS-GRPO 算法，通过状态等价性和强化学习训练评判模型 J4R，显著提升推理任务评估的准确性和一致性，7B 模型性能接近甚至超越许多大模型。"
---

> **Summary:** 本文提出 EIS-GRPO 算法，通过状态等价性和强化学习训练评判模型 J4R，显著提升推理任务评估的准确性和一致性，7B 模型性能接近甚至超越许多大模型。 

> **Keywords:** LLM, Reinforcement Learning, Evaluation, Reasoning, Bias Mitigation

**Authors:** Austin Xu, Yilun Zhou, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty

**Institution(s):** Salesforce AI Research


## Problem Background

随着大型语言模型（LLM）的发展，模型输出评估从耗时的人工评估转向自动评估，LLM-as-judge 模型作为生成式评估器在简单领域（如聊天质量）表现良好，但在推理密集型领域（如数学、逻辑推理）表现不佳，存在位置偏见和一致性问题，导致评估结果不稳定，论文旨在提升评判模型在复杂推理任务中的准确性和鲁棒性。

## Method

* **核心思想**：通过强化学习（RL）训练评判模型，利用状态等价性（state equivalence）对抗非实质性输入变换（如候选答案顺序），提升模型在推理任务中的一致性和准确性。
* **具体实现**：提出 Equivalent Initial State Group Relative Policy Optimization (EIS-GRPO) 算法，基于 Group Relative Policy Optimization (GRPO) 改进，步骤如下：
  * 对输入进行实质等价变换（如交换候选答案顺序），生成多个等价初始状态（transformed initial states）。
  * 在每个变换状态下采样一组响应（subgroups），形成多个子组。
  * 计算两部分优势（advantage）：全局优势（global advantage）基于所有子组的奖励均值和标准差，局部优势（local advantage）基于单个子组内部的奖励分布，两者结合用于策略更新。
  * 通过联合优势计算，模型学习到不同变换下的输入是等价的，从而减少位置偏见。
* **应用细节**：在评判任务中，变换主要为候选答案顺序，奖励机制包括判断奖励（judgment reward）和格式奖励（format reward），训练数据基于推理任务（如 MATH, ReClor）构建。
* **关键创新**：相比传统 GRPO，EIS-GRPO 引入状态等价性训练，不增加额外训练开销，同时显著提升一致性；训练了 J4R（Judge for Reasoning）模型，专注于推理评估。

## Experiment

* **有效性**：J4R-CJ-7B（7B 参数模型）在多个基准测试（如 PPE Best-of-K, JudgeBench, ReasoningJudgeBench）上表现优异，整体准确率达 50.35%，超越 GPT-4o（45.25%）和许多更大模型（如 RM-R1-14B 的 46.12%），接近 RM-R1-32B（51.80%）。
* **一致性提升**：EIS-GRPO 显著提高评判一致性，在 JudgeBench 上从 68.86% 提升至 81.14%，在 ReasoningJudgeBench 上从 68.58% 提升至 80.51%，有效缓解位置偏见。
* **实验设置**：实验覆盖多种推理任务（数学、多跳推理、领域特定推理、日常推理），数据来源多样（如 ARC-Challenge, ReClor, AIME），并通过双序评估确保一致性度量；新提出的 ReasoningJudgeBench 包含 1483 个样本，相比 JudgeBench（350 个）覆盖更广，设置合理。
* **计算效率**：在 FLOP-matched 评估中，J4R-CJ-7B 在等计算量下表现更优，与 RM-R1-32B 相比准确率从 54.29% 提升至 64.94%，显示小模型潜力。
* **局限性**：在非推理任务（如 HHH, LFQA）性能有所下降，表明领域特化可能影响泛化能力。

## Further Thoughts

EIS-GRPO 的状态等价性思想不仅适用于评判任务，还可能推广到其他需要鲁棒性的 NLP 任务，如问答系统或对话生成中对格式无关性的训练；此外，全局与局部优势结合的设计为多任务 RL 提供了新思路，值得探索其在其他领域的应用；J4R-CJ-7B 作为小模型接近大模型表现，提示通过精心设计训练方法，小模型在专业任务上可能实现高效替代。