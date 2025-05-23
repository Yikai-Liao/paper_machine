---
title: "Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning"
pubDatetime: 2025-05-20T11:22:34+00:00
slug: "2025-05-rlvr-distillation-reasoning"
type: "arxiv"
id: "2505.14216"
score: 0.6465436924126532
author: "grok-3-latest"
authors: ["Minwu Kim", "Anubhav Shrestha", "Safal Shrestha", "Aadim Nepal", "Keith Ross"]
tags: ["LLM", "Distillation", "Reasoning", "Reinforcement Learning", "Capability"]
institution: ["New York University Abu Dhabi"]
description: "本文通过对比 RLVR 和蒸馏，揭示 RLVR 因聚焦简单问题而仅提升准确率，蒸馏则需引入新知识才能提升能力，为语言模型推理训练提供深刻见解。"
---

> **Summary:** 本文通过对比 RLVR 和蒸馏，揭示 RLVR 因聚焦简单问题而仅提升准确率，蒸馏则需引入新知识才能提升能力，为语言模型推理训练提供深刻见解。 

> **Keywords:** LLM, Distillation, Reasoning, Reinforcement Learning, Capability

**Authors:** Minwu Kim, Anubhav Shrestha, Safal Shrestha, Aadim Nepal, Keith Ross

**Institution(s):** New York University Abu Dhabi


## Problem Background

大型语言模型（LLMs）在推理任务中表现出快速进步，但训练方法的效果存在差异：基于可验证奖励的强化学习（RLVR）能提升准确率（Accuracy），却无法提升模型能力（Capability，即输出分布中是否包含正确答案）；而蒸馏（Distillation）则在准确率和能力上均有提升。
论文旨在探究‘为什么’会出现这种现象，深入分析 RLVR 和蒸馏如何塑造模型的推理行为，解决的关键问题是揭示两种方法在准确率和能力提升上的机制差异。

## Method

*   **核心思想:** 通过对比 RLVR 和蒸馏对模型推理行为的影响，分析为何 RLVR 仅提升准确率而非能力，以及蒸馏提升能力的条件。
*   **RLVR 分析:** 研究基于 GRPO 等算法的 RLVR 训练机制，发现其倾向于优化较简单问题的准确率（通过策略梯度更新强化已有正确答案的概率），而对最难问题关注不足，导致能力无提升甚至下降。
*   **蒸馏分析:** 将教师模型响应拆分为推理模式（Reasoning Patterns）和领域知识（Domain Knowledge）两部分，通过控制变量实验设计：
    *   仅蒸馏推理模式：使用基础模型已知问题的教师响应进行监督微调（SFT），避免引入新知识。
    *   引入新知识：使用如 DeepSeek-R1 蒸馏模型，包含大量外部知识的响应进行训练。
    *   对比两者的准确率和能力变化，揭示提升机制。
*   **实验方法:** 使用 Qwen2.5-1.5B-Math 和 Qwen2.5-3B 作为基础模型，结合公开模型和自训练模型，在 MATH 和 AIME25 数据集上评估 pass@k 能力和准确率；此外，通过自蒸馏实验和定性分析（响应长度、反思关键词）进一步验证 RLVR 的响应质量变化。

## Experiment

*   **RLVR 效果:** 在 MATH500 测试集上，RLVR 将 Qwen2.5-1.5B-Math 的准确率从 60.6% 提升至 74.2%，但 pass@256 能力几乎不变（从 97.2% 微降至 97.0%），验证了其‘牺牲难题’特性，即对简单问题提升显著，对最难问题无帮助甚至退步。
*   **蒸馏效果:** 引入新知识的 DeepSeek 模型在 AIME25 上将 pass@256 能力从 56.7% 提升至 70.0%，准确率也有显著提升；仅蒸馏推理模式的模型准确率提升（MATH500 从 60% 至 68%），但能力无变化，与 RLVR 类似。
*   **其他发现:** 自蒸馏实验表明 RLVR 能生成更高质量响应（基础模型蒸馏 RL 响应后测试准确率提升 11.6%），但表面特征（如响应长度、反思关键词）并非质量可靠指标。
*   **实验设置评价:** 实验覆盖多个数据集和模型规模，pass@k 和准确率指标设计合理，控制变量有效分离推理模式和知识的影响；但局限于数学领域和小规模模型（1.5B、3B），泛化性待验证。

## Further Thoughts

论文揭示能力提升依赖于新知识引入，而非单纯推理模式优化，这启发我们可以在训练中针对性引入外部知识源（如领域特定数据）突破能力瓶颈；此外，RLVR 和蒸馏的资源使用差异（‘苹果与橙子’对比）提示可以探索混合策略，如结合 RLVR 的准确率优化与知识注入的能力扩展，设计更全面的训练框架。