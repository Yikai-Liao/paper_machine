---
title: "ExTrans: Multilingual Deep Reasoning Translation via Exemplar-Enhanced Reinforcement Learning"
pubDatetime: 2025-05-19T11:34:47+00:00
slug: "2025-05-extrans-reward-modeling"
type: "arxiv"
id: "2505.12996"
score: 0.7783943994097134
author: "grok-3-latest"
authors: ["Jiaan Wang", "Fandong Meng", "Jie Zhou"]
tags: ["LLM", "Reinforcement Learning", "Machine Translation", "Reasoning", "Multilingual"]
institution: ["Tencent Inc, Pattern Recognition Center, WeChat AI"]
description: "本文提出了一种示例增强的强化学习奖励建模方法，利用强大模型的双重角色提升翻译性能，并通过轻量级策略实现多语言能力迁移，在文学翻译任务中取得显著成果。"
---

> **Summary:** 本文提出了一种示例增强的强化学习奖励建模方法，利用强大模型的双重角色提升翻译性能，并通过轻量级策略实现多语言能力迁移，在文学翻译任务中取得显著成果。 

> **Keywords:** LLM, Reinforcement Learning, Machine Translation, Reasoning, Multilingual

**Authors:** Jiaan Wang, Fandong Meng, Jie Zhou

**Institution(s):** Tencent Inc, Pattern Recognition Center, WeChat AI


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）在复杂任务（如数学、编码）中表现出色，但其在神经机器翻译（Neural Machine Translation, MT）中的应用主要局限于高资源语言（如英语和中文），对其他语言的表现尚不明确；此外，强化学习（Reinforcement Learning, RL）在翻译中的奖励建模方法未充分发挥潜力，限制了翻译能力的进一步提升。
论文旨在通过改进奖励建模方法，增强LRMs在翻译任务中的表现，并将这种能力扩展到多语言场景。

## Method

*   **示例增强的奖励建模（Exemplar-Enhanced Reward Modeling）**：
    *   利用强大的LRM（如DeepSeek-R1）作为‘示例’（exemplar），为源句子生成高质量的翻译作为基准。
    *   在强化学习训练中，通过另一个强大模型（如DeepSeek-v3）比较策略模型（policy model）的翻译结果与示例翻译的质量，生成奖励信号，奖励值根据比较结果分为多个等级（1到5），以鼓励策略模型超越示例。
    *   同时结合格式奖励（确保输出格式正确）、思维奖励（评估推理过程质量）和辅助奖励（如CometKiwi分数），综合计算最终奖励值。
    *   这种方法充分利用了‘LLM作为评判者’（LLM-as-a-judge）和‘LLM作为示例’（LLM-as-an-exemplar）的双重能力，为策略模型设定更高的性能下限。
*   **轻量级多语言奖励建模（Lightweight Multilingual Reward Modeling）**：
    *   针对多语言翻译场景，设计低成本奖励机制：对于高资源语言方向（如英-中），采用完整的奖励建模；对于其他语言方向，仅通过正则表达式和语言检测工具验证生成格式和目标语言的正确性，避免直接评估翻译质量。
    *   这种方法旨在通过高资源语言方向的训练成果迁移到多语言场景，降低计算成本并规避LLM在低资源语言上的不准确性。
*   **训练流程**：
    *   使用Qwen2.5-7B-Instruct作为骨干模型，训练分为冷启动监督微调（Supervised Fine-Tuning, SFT）和强化学习两个阶段。
    *   冷启动SFT通过DeepSeek-R1生成带长链推理（Chain-of-Thought, CoT）的翻译数据，鼓励模型形成‘先思考后翻译’的模式；强化学习阶段采用GRPO算法优化策略模型。

## Experiment

*   **英-中文学翻译（ExTrans-7B）**：
    *   在MetaphorTrans等数据集上，ExTrans-7B显著优于之前的MT LRMs（如DeepTrans-7B）和强基准模型（如OpenAI-o1, DeepSeek-R1），在GRF（90.55）、GEA5（4.60）、GEA100（82.29）和CometKiwi（74.23）等指标上达到最优，相比DeepTrans-7B提升明显（例如GRF提升1.9%）。
    *   消融实验验证了示例增强奖励（r_trans）和CometKiwi奖励（r_cometk）的重要性，去除任一奖励均导致性能下降。
    *   实验设置合理，涵盖多个文学数据集（MetaphorTrans, O. Henry, Orbital），并通过GPT-4o和人类评估进一步验证效果。
*   **多语言翻译（mExTrans-7B）**：
    *   在11种语言（90个翻译方向）上，mExTrans-7B通过轻量级奖励建模实现能力迁移，性能在大多数方向上优于冷启动版本和部分基准（如QwQ-32B），例如在英-阿拉伯语方向GRF提升至81.10，但与o1-preview仍有差距。
    *   实验设计全面，数据量分配合理（英-中19K样本，其他方向仅50样本），体现了迁移能力的测试，但低资源语言上的表现仍需提升。
*   **成本与局限**：训练成本较高（RL训练耗费1K+ GPU小时），且多语言场景下与顶级模型的性能差距表明轻量级奖励机制的迁移效果有限。

## Further Thoughts

论文提出的示例增强奖励机制启发了我：是否可以通过动态选择示例模型（例如根据任务难度选择不同规模的模型）来优化奖励信号的精准性？此外，轻量级能力迁移策略也让我思考：在多语言或跨领域任务中，是否可以引入少量人工标注数据或跨语言知识图谱来校准奖励机制，进一步提升低资源场景的表现？