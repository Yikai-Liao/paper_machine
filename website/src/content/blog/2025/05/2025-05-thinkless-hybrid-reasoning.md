---
title: "Thinkless: LLM Learns When to Think"
pubDatetime: 2025-05-19T17:24:16+00:00
slug: "2025-05-thinkless-hybrid-reasoning"
type: "arxiv"
id: "2505.13379"
score: 0.857657365288996
author: "grok-3-latest"
authors: ["Gongfan Fang", "Xinyin Ma", "Xinchao Wang"]
tags: ["LLM", "Reasoning", "Reinforcement Learning", "Hybrid Reasoning", "Efficiency"]
institution: ["National University of Singapore"]
description: "本文提出 Thinkless 框架，通过解耦 GRPO 强化学习算法使大型语言模型根据任务复杂度和自身能力自适应选择推理模式，在显著降低计算成本的同时保持性能。"
---

> **Summary:** 本文提出 Thinkless 框架，通过解耦 GRPO 强化学习算法使大型语言模型根据任务复杂度和自身能力自适应选择推理模式，在显著降低计算成本的同时保持性能。 

> **Keywords:** LLM, Reasoning, Reinforcement Learning, Hybrid Reasoning, Efficiency

**Authors:** Gongfan Fang, Xinyin Ma, Xinchao Wang

**Institution(s):** National University of Singapore


## Problem Background

推理型语言模型（Reasoning Language Models）在处理复杂任务时表现出色，但对所有查询统一采用链式推理（Chain-of-Thought）会导致不必要的计算开销，尤其是在简单问题上，产生冗余 token、增加内存占用和延迟。
作者提出一个开放性问题：LLM 是否能学会‘何时思考’，即根据任务复杂度和模型能力动态选择短格式（简洁回答）或长格式（详细推理）的响应方式，以在性能和效率间取得平衡。

## Method

* **核心思想：** 提出 Thinkless 框架，通过强化学习使 LLM 自主学习根据任务复杂度和自身能力选择合适的推理模式（短格式或长格式），避免手动启发式规则的局限性。
* **实现步骤：**
  1. **蒸馏预热阶段（Distillation for Warm-up）：** 通过监督微调（Supervised Fine-Tuning, SFT），利用两个专家模型（一个擅长详细推理，另一个擅长简洁回答）生成配对数据集，训练目标模型在控制 token（如 `<think>` 和 `<short>`）引导下生成对应风格的响应，确保模型具备生成两种推理模式的基础能力。
  2. **强化学习阶段（Reinforcement Learning with Decoupled GRPO）：** 引入解耦组相对策略优化（Decoupled Group Relative Policy Optimization, DeGRPO）算法，将混合推理目标分解为两个部分：
     - **模式选择（Mode Selection）：** 优化控制 token 的选择概率，根据任务复杂度和模型能力决定推理模式。
     - **响应准确性提升（Accuracy Improvement）：** 优化后续响应 token 的生成质量，确保回答正确性。
     DeGRPO 通过独立归一化和加权参数 α 平衡控制 token 和响应 token 的梯度更新，避免传统 GRPO 中的模式崩塌（Mode Collapse）问题。奖励函数设计偏向短格式正确回答，以鼓励高效推理。
* **关键创新：** DeGRPO 算法通过解耦优化目标，解决了控制 token 和响应 token 数量不平衡导致的训练不稳定问题，确保模型能动态适应不同任务需求。

## Experiment

* **有效性：** 在多个数学推理数据集（如 AIME、Minerva Algebra、MATH-500、GSM8K）上，Thinkless 显著减少长格式推理的使用比例（部分数据集减少 50%-90%），例如在 Minerva Algebra 上仅 25.88% 查询使用长格式，token 使用量降至原来的三分之一，性能仅下降 1%。
* **自适应性：** 在复杂数据集（如 AIME）上，模型倾向更多使用长格式推理，而在简单数据集（如 GSM8K）上更多使用短格式，体现了根据任务难度动态调整的能力。
* **对比优势：** 与基线模型（纯推理模型和简洁模型）、模型合并（Merging）和路由方法相比，Thinkless 在效率和性能权衡上表现更优，尤其避免了独立路由模型对模型能力感知不足的局限性。
* **训练动态：** DeGRPO 训练呈现 U 型学习曲线，初期偏向长格式以保证准确性，后期随着短格式准确性提升逐渐增加其比例，而传统 GRPO 因梯度不平衡导致模式崩塌，验证了解耦优化的必要性。
* **实验设置合理性：** 实验覆盖不同难度数据集，评估指标包括准确率（Pass@1）和 token 使用量，设置了多种对比方法，较为全面；但数据集主要集中于数学领域，缺乏更广泛领域的验证，可能限制普适性。

## Further Thoughts

Thinkless 的自适应推理思想可扩展至代码生成或多模态任务，通过类似框架让模型在不同任务间动态分配计算资源；控制 token 的设计可引入更多层次（如 `<medium>`）或结合用户偏好（如延迟容忍度）实现更细粒度控制；DeGRPO 解耦优化的思路或可应用于其他多目标强化学习任务；此外，预热阶段可通过先进的模型合并技术（如 LoRA）或多任务学习提升初始模型质量，进一步增强后续强化学习效果。