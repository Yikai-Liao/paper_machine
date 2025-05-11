---
title: "Scalable Chain of Thoughts via Elastic Reasoning"
pubDatetime: 2025-05-08T15:01:06+00:00
slug: "2025-05-elastic-reasoning-budget"
type: "arxiv"
id: "2505.05315"
score: 0.6897428655315844
author: "grok-3-latest"
authors: ["Yuhui Xu", "Hanze Dong", "Lei Wang", "Doyen Sahoo", "Junnan Li", "Caiming Xiong"]
tags: ["LLM", "Reasoning", "Test Time Scaling", "Post-Training", "RLHF"]
institution: ["Salesforce AI Research"]
description: "本文提出 Elastic Reasoning 框架，通过将推理分为思考和解决方案两阶段并结合预算约束训练，使大型推理模型在严格资源限制下仍能高效推理，同时降低训练成本并提升泛化能力。"
---

> **Summary:** 本文提出 Elastic Reasoning 框架，通过将推理分为思考和解决方案两阶段并结合预算约束训练，使大型推理模型在严格资源限制下仍能高效推理，同时降低训练成本并提升泛化能力。 

> **Keywords:** LLM, Reasoning, Test Time Scaling, Post-Training, RLHF

**Authors:** Yuhui Xu, Hanze Dong, Lei Wang, Doyen Sahoo, Junnan Li, Caiming Xiong

**Institution(s):** Salesforce AI Research


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）通过生成长篇推理链（Chain of Thought, CoT）在复杂任务中表现出色，但其输出长度不受控制，导致在实际部署中难以满足严格的推理时间预算（token、延迟或计算资源）限制。
现有方法在性能保持和训练成本上存在不足，特别是在预算受限时，模型性能显著下降，因此需要在资源约束下实现有效的长篇推理。

## Method

*   **核心思想:** 提出 Elastic Reasoning 框架，通过将推理过程分为‘思考（Thinking）’和‘解决方案（Solution）’两个阶段，并为每个阶段独立分配预算，实现推理链的可扩展控制。
*   **具体实现:**
    *   **Separate Budgeting:** 将总预算 *c* 分为思考阶段预算 *t* 和解决方案阶段预算 *s*（*c = t + s*）。在推理时，若思考阶段未用尽预算 *t* 就自然结束（即生成结束标记如 `</think>`），则直接进入解决方案阶段；若预算 *t* 用尽，则强制插入结束标记并进入解决方案阶段，确保解决方案部分不被截断。
    *   **Budget-Constrained Rollout:** 在训练时，结合 GRPO 强化学习算法，模拟推理时的预算限制，训练模型在思考阶段被截断的情况下仍能生成高质量的解决方案。训练使用固定预算（如 1K+1K），但模型能泛化到不同预算配置。
*   **关键优势:** 该方法不仅控制了推理长度，还通过强化学习提升了模型在预算限制下的适应性，训练成本低（数学任务仅需 200 步），且推理时能灵活调整预算分配。

## Experiment

*   **有效性:** 在严格预算限制下，Elastic Reasoning 显著优于基线方法（如 S1 和 L1-Exact），并与 L1-Max 性能相当。例如，在 AIME2024 数据集上，E1-Math-1.5B 模型达到 35.0% 准确率，相比 L1-Max 的 27.1% 和 L1-Exact 的 24.2% 有明显提升；在编程任务中，E1-Code-14B 模型随预算增加性能稳定提升。
*   **效率:** 训练成本大幅降低，数学任务仅需 200 步，而 L1-Max 需要 820 步；推理时 token 使用量减少 30%-37.4%（如 AIME2024 减少 32.1%）。
*   **泛化性:** 模型在训练时使用固定预算（1K+1K），测试时能适应不同预算配置，表现出鲁棒性。
*   **实验设置:** 实验覆盖数学（AIME, MATH500 等）和编程（LiveCodeBench, Codeforces 等）多个数据集，预算配置多样，验证了方法的广泛适用性，但未深入探讨任务复杂度对预算分配的影响。

## Further Thoughts

Elastic Reasoning 的 Separate Budgeting 概念启发了我思考是否可以基于任务难度设计自适应预算分配策略，例如为复杂任务动态增加思考预算；此外，Budget-Constrained Rollout 的训练方式是否可应用于对话系统或多模态任务中的资源控制；进一步地，是否可以通过多阶段预算分配（不仅仅是两阶段），实现更细粒度的推理过程优化。