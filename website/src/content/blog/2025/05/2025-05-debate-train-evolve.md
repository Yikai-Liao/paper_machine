---
title: "DEBATE, TRAIN, EVOLVE: Self Evolution of Language Model Reasoning"
pubDatetime: 2025-05-21T16:40:12+00:00
slug: "2025-05-debate-train-evolve"
type: "arxiv"
id: "2505.15734"
score: 0.7956271958867462
author: "grok-3-latest"
authors: ["Gaurav Srivastava", "Zhenyu Bi", "Meng Lu", "Xuan Wang"]
tags: ["LLM", "Multi-Agent Debate", "Self-Evolution", "Reasoning", "Reinforcement Learning"]
institution: ["Virginia Tech"]
description: "本文提出 DEBATE, TRAIN, EVOLVE 框架，通过多智能体辩论轨迹和自监督强化学习，实现语言模型推理能力的自主进化，在保持单模型效率的同时显著提升性能。"
---

> **Summary:** 本文提出 DEBATE, TRAIN, EVOLVE 框架，通过多智能体辩论轨迹和自监督强化学习，实现语言模型推理能力的自主进化，在保持单模型效率的同时显著提升性能。 

> **Keywords:** LLM, Multi-Agent Debate, Self-Evolution, Reasoning, Reinforcement Learning

**Authors:** Gaurav Srivastava, Zhenyu Bi, Meng Lu, Xuan Wang

**Institution(s):** Virginia Tech


## Problem Background

大型语言模型（LLMs）在推理能力上的提升主要依赖大规模数据集训练，但数据饱和问题日益凸显，额外数据的边际效益递减。
作者提出关键问题：如何在不依赖外部监督或额外数据的情况下，让模型自主提升推理能力？
现有自进化方法常受限于单模型确认偏差和推理多样性不足，而多智能体辩论（Multi-Agent Debate, MAD）虽有效，但作为推理时技术需多模型同时运行，计算开销和延迟高，不适合大规模部署。
因此，本文旨在通过利用辩论轨迹训练单个模型，实现高效推理能力提升。

## Method

*   **核心思想:** 提出 DEBATE, TRAIN, EVOLVE (DTE) 框架，通过多智能体辩论生成高质量推理轨迹，结合自监督强化学习对单个模型进行微调，实现推理能力自主进化，同时保持单模型推理的低成本。
*   **具体步骤:**
    *   **Reflect-Critique-Refine (RCR) 提示策略:** 针对传统 MAD 提示中的‘马屁精’行为和冗长偏见，设计结构化提示，强制智能体反思自身答案潜在错误，批判至少两个同伴推理缺陷，并仅在提供新推理步骤时修改答案，以提升辩论质量。
    *   **DTE 框架三阶段:**
        *   **Debate:** 多个智能体基于 RCR 提示进行辩论，生成推理轨迹，直到达成共识或达到最大轮数。
        *   **Train:** 从辩论轨迹中提取共识答案和高质量推理步骤，使用 Group Relative Policy Optimization (GRPO) 微调单个模型；GRPO 通过设计多组件奖励函数（答案正确性、格式一致性、简洁性）优化输出，并用 KL 散度约束避免灾难性遗忘。
        *   **Evolve:** 进化后的模型替换原始版本，进入下一轮迭代，直到验证奖励稳定或达到最大迭代次数。
*   **关键创新:** 不依赖真实标签，完全基于自生成辩论数据训练；RCR 提示提升辩论质量；GRPO 平衡探索与稳定性，确保进化过程高效且稳定。

## Experiment

*   **有效性:** 在五个推理基准数据集（GSM8K, GSM-Plus, ARC-Easy, ARC-Challenge, CommonsenseQA）上测试六种开源模型，DTE 框架在 GSM-Plus 上平均准确率提升 8.92%，在其他数据集上平均提升 5.8%，显著增强了推理能力。
*   **优越性:** 进化后的单模型性能接近甚至超过多智能体辩论（如 GSM-Plus 上平均提升 2.38 个百分点），恢复了单模型推理的低延迟和低计算成本；RCR 提示相比传统 MAD 提示平均提升 1.9-3.7 个百分点，并将‘马屁精’率从 0.28 降至 0.13。
*   **跨领域泛化:** 在未见数据集上表现出色（如 GSM8K 训练模型在 GSM-Plus 提升 5.8%），表明 DTE 捕捉了通用推理能力，而非特定模式。
*   **实验设置合理性:** 覆盖多种模型规模（1.5B-14B 参数）和任务类型，设置单模型和传统 MAD 基线，进行多维度消融研究（如 RCR 效果、代理数量、GRPO vs 其他方法），验证全面。
*   **局限性:** 小模型（<3B 参数）在多轮进化后易出现灾难性遗忘，需降低采样温度缓解；第二轮进化收益有限，部分模型性能下降，迭代稳定性待优化。

## Further Thoughts

DTE 框架展示了多智能体交互生成的数据可有效提升单模型能力，这启发我们是否能在其他任务（如对话系统、代码生成）中利用类似机制，通过‘群体智慧’增强个体性能；RCR 提示通过结构化指令提升辩论质量，提示我们在提示设计中可更注重任务分解和约束，是否能进一步设计动态提示根据辩论进程调整指令？此外，小模型迭代遗忘问题是否可通过混合训练或‘记忆保护’机制缓解？