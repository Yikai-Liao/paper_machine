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
description: "本文提出 DTE 框架，通过多智能体辩论和 RCR 提示策略生成高质量推理轨迹，利用 GRPO 优化训练单个语言模型，实现自主推理能力提升和高效单模型推理。"
---

> **Summary:** 本文提出 DTE 框架，通过多智能体辩论和 RCR 提示策略生成高质量推理轨迹，利用 GRPO 优化训练单个语言模型，实现自主推理能力提升和高效单模型推理。 

> **Keywords:** LLM, Multi-Agent Debate, Self-Evolution, Reasoning, Reinforcement Learning

**Authors:** Gaurav Srivastava, Zhenyu Bi, Meng Lu, Xuan Wang

**Institution(s):** Virginia Tech


## Problem Background

大型语言模型（LLMs）在推理能力上的进步依赖于大规模数据集训练，但随着数据饱和点的临近，单纯增加数据已不足以进一步提升性能，同时现有自进化方法存在确认偏差和推理多样性不足的问题，多智能体辩论（MAD）虽有效但计算开销高；论文旨在探索如何在无外部监督的情况下，通过整合多智能体辩论的优点，实现单模型的高效自主推理能力提升。

## Method

*   **核心思想:** 提出‘DEBATE, TRAIN, EVOLVE’（DTE）框架，通过多智能体辩论生成高质量推理轨迹，训练单个语言模型实现自主进化，同时保持推理时的高效性。
*   **具体实现:** 
    *   **RCR 提示策略（Reflect-Critique-Refine）:** 设计一种新的提示方法，要求智能体反思自身答案的潜在错误、批评同伴推理中的具体缺陷，并改进答案，从而提高辩论质量，减少‘马屁精行为’（sycophancy）和冗长偏见（verbosity bias）。
    *   **DTE 框架流程:** 首先通过多智能体辩论生成推理轨迹，提取共识答案和关键推理步骤；然后利用这些数据，通过 Group Relative Policy Optimization (GRPO) 微调单个模型；进化后的模型替换原始模型，循环进行，直到性能收敛或达到最大迭代次数。
    *   **GRPO 优化细节:** 设计多重奖励函数，包括答案正确性、格式一致性和输出简洁性，同时通过 KL 散度约束避免灾难性遗忘，确保模型在进化过程中保持稳定性。
*   **关键创新:** 不依赖真实标签（ground truth），通过多智能体交互克服单模型自反馈的局限性，将多智能体辩论的计算开销转移到训练阶段，推理时仅需单模型前向传播。

## Experiment

*   **有效性:** DTE 框架在 GSM-Plus 数据集上平均提升了 8.92% 的准确率，尤其对较小模型（如 Qwen-1.5B）提升显著（+13.92%）；在其他四个推理基准数据集上表现出跨领域泛化能力，平均提升 5.8%。
*   **对比分析:** 相比原始单模型，DTE 显著提升了推理性能；相比传统多智能体辩论（MAD），DTE 通过单模型推理降低了计算开销；RCR 提示策略相比传统 MAD 提示在 GSM8K 和 GSM-Plus 上分别提升了 1.9% 和 3.7% 的准确率，并将‘马屁精行为’率从 0.28 降至 0.13。
*   **实验设置合理性:** 实验覆盖了多种模型规模（1.5B 到 14B 参数）和五个推理基准数据集，测试了跨领域泛化能力，并通过消融研究分析了 RCR 提示、GRPO 优化、代理数量、数据选择策略等因素的影响，设置较为全面。
*   **局限性与成本:** 小模型（<3B 参数）在多轮进化后易出现灾难性遗忘，需通过降低采样温度缓解；训练成本仍高于标准单模型微调，但相比传统 MAD 推理时成本更低。

## Further Thoughts

多智能体辩论生成多样化推理轨迹的思路启发了我，是否可以通过设计异构智能体（不同架构、不同训练数据的模型）进一步提升辩论质量，例如结合擅长数学推理和常识推理的模型生成更全面的推理轨迹？此外，是否可以根据任务难度动态调整辩论轮数或智能体数量，以平衡性能和计算成本？奖励函数是否可以引入更多维度，如推理逻辑的连贯性或创新性，以进一步优化模型进化？