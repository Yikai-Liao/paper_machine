---
title: "The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning"
pubDatetime: 2025-05-21T05:39:11+00:00
slug: "2025-05-entropy-minimization-llm"
type: "arxiv"
id: "2505.15134"
score: 0.7936242106437946
author: "grok-3-latest"
authors: ["Shivam Agarwal", "Zimin Zhang", "Lifan Yuan", "Jiawei Han", "Hao Peng"]
tags: ["LLM", "Entropy Minimization", "Reasoning", "Post-Training", "Inference Scaling"]
institution: ["University of Illinois Urbana-Champaign"]
description: "本文提出熵最小化方法，通过无监督微调、强化学习和推理时 logits 调整，显著提升大型语言模型在复杂推理任务上的性能，无需标注数据或参数更新。"
---

> **Summary:** 本文提出熵最小化方法，通过无监督微调、强化学习和推理时 logits 调整，显著提升大型语言模型在复杂推理任务上的性能，无需标注数据或参数更新。 

> **Keywords:** LLM, Entropy Minimization, Reasoning, Post-Training, Inference Scaling

**Authors:** Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, Hao Peng

**Institution(s):** University of Illinois Urbana-Champaign


## Problem Background

大型语言模型（LLMs）在复杂推理任务（如数学、物理和编码）上表现出强大潜力，但传统方法依赖标注数据进行监督微调或强化学习。
本文探索一个核心问题：能否通过熵最小化（Entropy Minimization, EM），即让模型对其最自信的输出分配更多概率质量，在无需标注数据或外部监督的情况下显著提升 LLMs 的推理性能？
这一研究旨在挖掘预训练模型中未被充分利用的推理能力，挑战传统依赖监督学习的范式。

## Method

*   **核心思想:** 通过减少模型输出分布的熵（Entropy），强化模型对其自信预测的倾向，从而提升推理任务性能，无需标注数据。
*   **具体方法:** 提出了三种熵最小化策略，分别应用于后训练和推理阶段：
    *   **EM-FT (Entropy Minimization through Finetuning):** 一种无监督微调方法，直接最小化 token 级别的熵。给定输入提示，从模型采样输出（无标注），计算 token 级熵并通过梯度下降优化模型参数以减少熵，类似于监督微调但不依赖标注数据。
    *   **EM-RL (Entropy Minimization with Reinforcement Learning):** 使用强化学习，以负熵作为唯一奖励信号进行优化。分为两种变体：EM-RL-sequence（基于轨迹级熵，奖励高概率完整序列）和 EM-RL-token（基于 token 级熵，奖励每步更确定的分布）。通过策略梯度方法优化，辅以 KL 正则化避免偏离基础模型过远。
    *   **EM-INF (Entropy Minimization at Inference Time):** 在推理时通过调整 logits 减少熵，无需更新模型参数。将模型最后一层 logits 视为可优化参数，通过梯度下降最小化输出分布熵（设置最小熵阈值避免过度优化），然后基于优化后分布进行采样解码，计算开销低且即插即用。

## Experiment

*   **有效性:** EM-FT 在 Qwen2.5-7B 上使用无标注数据，平均提升数学和编码任务性能约 8%，在 Minerva 和 LeetCode 上甚至超过使用 60K 标注数据的 GRPO 和 RLOO（例如 Minerva 准确率 33.1% vs. 25.0%）。
*   **优越性:** EM-RL 平均提升 11%，在多个任务上与监督方法竞争（例如 AMC 上 EM-RL-token 达到 57.8%，与 RLOO 持平）；EM-INF 在推理时平均提升 3%，在高不确定性任务如 AIME 和 SciCode 上表现突出，Qwen2.5-32B 在 SciCode 主问题上超越 GPT-4o（10.7% vs. 9.2%），且效率是自一致性方法的 3 倍。
*   **实验设置:** 实验覆盖数学、编码、物理等多领域任务，使用 Qwen 系列和 Llama-3.1 模型，基准数据集（如 MATH、LeetCode、SciCode）具有代表性，训练数据量充足（60K 提示），并探讨了方法局限性（如在 Llama-3.1 和价值观对齐任务上效果不佳），设计全面合理。
*   **开销:** EM-FT 和 EM-RL 需额外训练计算（FLOPs 分别为 1.01e17 和 13e17），而 EM-INF 仅在推理时操作，计算开销低（与常规解码相当）。

## Further Thoughts

熵最小化作为无监督方法，能有效挖掘预训练模型的推理能力，尤其是在推理时调整 logits（EM-INF）无需训练即可提升性能。这启发我思考：是否可以设计自适应熵调整机制，根据任务不确定性动态选择熵最小化强度，例如在高不确定性任务上降低熵阈值鼓励探索，而在高自信任务上加强熵最小化强化确定性？此外，熵最小化是否可与其他无监督策略（如自一致性）结合，形成更强大的推理时方法？