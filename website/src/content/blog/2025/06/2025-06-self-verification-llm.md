---
title: "Incentivizing LLMs to Self-Verify Their Answers"
pubDatetime: 2025-06-02T06:54:29+00:00
slug: "2025-06-self-verification-llm"
type: "arxiv"
id: "2506.01369"
score: 0.8884275307701578
author: "grok-3-latest"
authors: ["Fuxiang Zhang", "Jiacheng Xu", "Chaojie Wang", "Ce Cui", "Yang Liu", "Bo An"]
tags: ["LLM", "Reasoning", "Reinforcement Learning", "Test-Time Scaling", "Self-Verification"]
institution: ["Nanyang Technological University, Singapore", "Skywork AI, Singapore"]
description: "本文提出自验证框架，通过强化学习统一大型语言模型的答案生成和验证能力，显著提升数学推理任务的后训练和测试时扩展性能，同时解决分布偏移问题。"
---

> **Summary:** 本文提出自验证框架，通过强化学习统一大型语言模型的答案生成和验证能力，显著提升数学推理任务的后训练和测试时扩展性能，同时解决分布偏移问题。 

> **Keywords:** LLM, Reasoning, Reinforcement Learning, Test-Time Scaling, Self-Verification

**Authors:** Fuxiang Zhang, Jiacheng Xu, Chaojie Wang, Ce Cui, Yang Liu, Bo An

**Institution(s):** Nanyang Technological University, Singapore, Skywork AI, Singapore


## Problem Background

大型语言模型（LLMs）在复杂推理任务中表现出色，但后训练（post-training）和测试时扩展（test-time scaling）方法的结合效果有限。
由于后训练模型针对特定任务调优，其生成分布与通用外部奖励模型存在分布差异（distribution shift），导致测试时扩展性能受限。
论文旨在弥合这一差距，提出一种无需外部验证器即可提升模型推理性能的框架。

## Method

*   **核心思想:** 提出自验证（self-verification）框架，通过单一强化学习（RL）过程统一答案生成和验证能力，避免分布偏移问题。
*   **训练过程:** 
    *   采用 Group Relative Policy Optimization (GRPO) 算法，同时训练模型解决数学推理问题和验证生成的解决方案。
    *   引入策略对齐缓冲区（policy-aligned buffer），存储最近生成的解决方案，确保验证数据与当前模型输出分布一致，缓解早期训练阶段无效数据的干扰。
    *   设计动态验证奖励（dynamic verification reward），根据验证任务难度调整奖励信号，解决正确解决方案主导导致的数据不平衡问题，激励模型关注困难验证案例。
*   **推理过程:** 
    *   在测试时，模型生成多个候选答案，并利用自身验证能力计算每个答案的验证分数。
    *   通过加权答案聚合（weighted answer aggregation）选择最终答案，平衡共识和置信度，无需外部奖励模型。
*   **优势:** 统一模型设计提升了计算效率，减少了推理时资源开销，同时保持了后训练和测试时扩展的协同性。

## Experiment

*   **后训练效果:** 自验证模型在数学推理基准（如 MATH500, AIME24）上显著优于基线模型和标准 GRPO 模型，例如 Self-Verification-Qwen-7B 在 MATH500 上得分从 62.00 提升至 83.60，表明自验证增强了问题解决能力。
*   **验证能力:** 自验证模型在验证自身解决方案时表现优异，准确率和 F1 分数接近甚至超过商业模型（如 GPT-4o），例如 Self-Verification-Qwen-7B 在 MATH500 验证准确率达 87.20%。
*   **测试时扩展:** 自验证方法在测试时扩展中表现最佳，例如 Self-Verification-R1-1.5B 在 MATH500 上得分达 93.60，优于 self-consistency 和 best-of-N 等方法，证明了其解决分布偏移问题的有效性。
*   **效率分析:** 自验证方法仅需部署单一模型，验证任务平均 token 使用量仅为问题解决任务的 24%-35%，时间成本低于依赖外部奖励模型的方法（如 beam search, DVTS）。
*   **实验设置合理性:** 实验覆盖多个基准数据集、不同模型规模和上下文长度，数据重复采样（如 AIME24 重复 10 次）增强了结果可靠性，但未探讨非数学推理任务的适用性。

## Further Thoughts

自验证框架通过统一生成和验证任务展示了任务协同训练的潜力，启发我们思考是否可以将类似方法扩展到其他领域（如代码生成或多模态任务），通过设计领域特定的奖励函数和缓冲机制实现性能提升；此外，动态奖励机制是否可以结合自适应学习率或多目标优化进一步改进训练稳定性？