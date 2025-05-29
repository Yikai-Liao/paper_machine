---
title: "Temporal Sampling for Forgotten Reasoning in LLMs"
pubDatetime: 2025-05-26T16:39:52+00:00
slug: "2025-05-temporal-sampling-reasoning"
type: "arxiv"
id: "2505.20196"
score: 0.9116985500431195
author: "grok-3-latest"
authors: ["Yuetai Li", "Zhangchen Xu", "Fengqing Jiang", "Bhaskar Ramasubramanian", "Luyao Niu", "Bill Yuchen Lin", "Xiang Yue", "Radha Poovendran"]
tags: ["LLM", "Reasoning", "Sampling", "Training Dynamics", "Inference Scaling"]
institution: ["University of Washington", "Carnegie Mellon University", "Western Washington University"]
description: "本文提出 Temporal Sampling 方法，通过在推理时从多个训练检查点采样，恢复大型语言模型中被遗忘的推理能力，显著提升了推理任务性能。"
---

> **Summary:** 本文提出 Temporal Sampling 方法，通过在推理时从多个训练检查点采样，恢复大型语言模型中被遗忘的推理能力，显著提升了推理任务性能。 

> **Keywords:** LLM, Reasoning, Sampling, Training Dynamics, Inference Scaling

**Authors:** Yuetai Li, Zhangchen Xu, Fengqing Jiang, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, Xiang Yue, Radha Poovendran

**Institution(s):** University of Washington, Carnegie Mellon University, Western Washington University


## Problem Background

大型语言模型（LLMs）在微调过程中会‘忘记’之前正确解答的问题答案，这一现象被称为‘Temporal Forgetting’（时间遗忘），研究发现6.4%到56.1%的最终错误答案在训练的某个中间检查点曾被正确解答，揭示了当前仅基于最终模型评估方法的局限性，论文旨在恢复这些被遗忘的推理能力并提升模型性能。

## Method

*   **核心思想:** 提出‘Temporal Sampling’（时间采样），一种在推理时利用训练过程中多个检查点的解码策略，通过训练动态（training dynamics）作为答案多样性的来源，恢复被遗忘的推理能力，而无需重新训练或集成多个模型。
*   **具体实现:** 
    *   将采样预算（sampling budget）分配到训练过程中的 *t* 个检查点上，采用轮询（round-robin）方式，从最近的 *t* 个检查点循环生成 *k* 个答案。
    *   例如，若有8个检查点，采样时依次从最新到最早的检查点各生成一个答案，循环进行，直到达到 *k* 个样本。
    *   引入新指标 *Pass@k|t*，衡量从 *t* 个检查点采样 *k* 次时至少获得一个正确答案的概率，并提供无偏估计方法。
*   **存储优化:** 将方法扩展到 LoRA（Low-Rank Adaptation）微调模型，仅保存低秩适配器权重而非完整模型参数，大幅降低存储成本。
*   **优势:** 不需要修改模型参数或额外训练，仅通过推理时调整采样策略即可实现性能提升，具有高实用性和计算效率。

## Experiment

*   **有效性:** Temporal Sampling 在多个推理基准（如 AIME2024, AMC, AIME2025）上显著提升性能，相比仅使用最终检查点的基线，*Pass@k* 指标提升了4到19个百分点，例如 Qwen2.5-7B 在 AIME2024 上使用 *t=8* 检查点时，*Pass@64|8* 比基线高出19个百分点。
*   **多样性指标:** 在多数投票（Majority@ *k*）和最佳选择（Best-of-N）指标上也有一致改进，例如在 AIME2024 上，*Maj@64|8* 比基线高出8个百分点，Best-of-N 提升7个百分点。
*   **效率与存储优化:** 结合 LoRA 微调后，Temporal Sampling 仍保持性能提升，同时大幅降低存储成本。
*   **对比分析:** 与混合模型（Mixture of Models, MoM）相比，Temporal Sampling 在相同计算预算下表现更优，例如在 AMC 上，*Maj@64* 比 MoM 高出9个百分点。
*   **实验设置合理性:** 实验覆盖多个基准、模型规模（如 Qwen2.5-1.5B, 7B）和微调方法（SFT, RL），采样参数（如温度0.6, top-p 0.95）设置合理，检查点数量（8个）足以捕捉训练动态，数据表明方法有效且提升显著，但未探讨不同采样分配策略的影响。

## Further Thoughts

模型的真实能力可能不体现在单一参数快照（最终模型）中，而是在整个训练动态（temporal diversity）中，这一思想挑战了传统‘最终模型至上’的评估范式，启发我们重新思考模型能力的定义和评估方式；未来可以探索动态选择检查点的策略，或将 Temporal Sampling 与其他推理时扩展方法（如 Tree-of-Thoughts）结合，进一步提升答案多样性和质量。