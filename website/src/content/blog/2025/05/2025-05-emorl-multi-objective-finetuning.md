---
title: "EMORL: Ensemble Multi-Objective Reinforcement Learning for Efficient and Flexible LLM Fine-Tuning"
pubDatetime: 2025-05-05T11:30:46+00:00
slug: "2025-05-emorl-multi-objective-finetuning"
type: "arxiv"
id: "2505.02579"
score: 0.7033596405616602
author: "grok-3-latest"
authors: ["Lingxiao Kong", "Cong Yang", "Susanne Neufang", "Oya Deniz Beyan", "Zeyd Boukhers"]
tags: ["LLM", "Ensemble Learning", "Multi-Objective Optimization", "Reinforcement Learning", "Fine-Tuning"]
institution: ["Fraunhofer Institute for Applied Information Technology FIT", "Soochow University", "University Hospital of Cologne"]
description: "EMORL 框架通过集成学习和隐藏状态聚合，为多目标 LLM 微调提供了一种高效、灵活且可解释的方法，在资源消耗和稳定性上显著优于传统方法，同时保持了相当的性能。"
---

> **Summary:** EMORL 框架通过集成学习和隐藏状态聚合，为多目标 LLM 微调提供了一种高效、灵活且可解释的方法，在资源消耗和稳定性上显著优于传统方法，同时保持了相当的性能。 

> **Keywords:** LLM, Ensemble Learning, Multi-Objective Optimization, Reinforcement Learning, Fine-Tuning

**Authors:** Lingxiao Kong, Cong Yang, Susanne Neufang, Oya Deniz Beyan, Zeyd Boukhers

**Institution(s):** Fraunhofer Institute for Applied Information Technology FIT, Soochow University, University Hospital of Cologne


## Problem Background

大型语言模型（LLM）在多目标任务中的微调面临训练效率低、目标平衡困难、可扩展性差和结果可解释性不足的挑战。
本文以辅导反思生成任务为背景，旨在生成同时具备反思性、共情性和流畅性的回应，解决传统强化学习（RL）方法在多目标优化中的收敛速度慢、训练不稳定和性能折衷问题。

## Method

*   **核心思想**：提出 EMORL（Ensemble Multi-Objective Reinforcement Learning）框架，通过集成学习将多目标优化分解为单目标优化，并通过聚合阶段优化组合，提升微调效率和灵活性。
*   **具体实现**：
    *   **训练阶段**：为每个目标（如反思、共情、流畅性）独立微调一个模型，使用强化学习算法（如 Self-Critical Sequence Training, SCST）结合 LoRA（Low-Rank Adaptation）技术减少参数更新负担，确保高效训练。
    *   **聚合阶段**：创新性地采用隐藏状态层级聚合（hidden-state level aggregation），将各单目标模型的最后隐藏状态（last hidden states）通过线性加权组合，融合不同目标的上下文信息，避免了参数层级或逻辑层级聚合中语义不一致的问题。
    *   **权重优化**：设计分层网格搜索（hierarchical grid search）算法，以 O(3^d * log_2 N) 的计算复杂度高效搜索最佳加权组合，相比标准网格搜索（O(N^d)）大幅减少计算成本。
*   **关键优势**：通过并行训练单目标模型避免多目标同时优化的复杂性，模块化设计支持新增目标，聚合阶段的可视化增强了结果的可解释性。

## Experiment

*   **效率提升**：EMORL 在资源消耗上表现突出，数据消耗为 17,529 ± 1,650 个数据点，训练时间为 6,573 ± 147.43 秒，相比传统单策略方法（如 Uniform Weighted 和 DynaOpt）减少约 0.5 倍资源消耗，稳定性也更高（数据和时间波动较小）。
*   **性能表现**：在 PAIR 数据集上，EMORL 平均得分为 0.7907，与单策略方法（Uniform Weighted 0.8397, DynaOpt 0.8254）相当，优于参数层级聚合的 Model Soups（0.6982）；在 Psych8k 数据集上，反思得分达 0.9784，为所有模型最高。
*   **多样性与可解释性**：EMORL 的 Diversity-2 得分为 0.6516，远高于其他微调模型（0.35-0.43），生成文本多样性更强；通过分层网格搜索可视化，清晰展示各目标贡献（如反思权重约 0.8，流畅性约 0.05），增强了可解释性。
*   **实验设置合理性**：实验涵盖 PAIR 和 Psych8k 两个数据集，结合自动与人工评估，对比多种基线（T5-base、Uniform Weighted、DynaOpt、Model Soups），设置较为全面；但使用小型模型（T5-base）限制了生成质量，未来需在大模型上验证。

## Further Thoughts

隐藏状态层级聚合为多目标 NLP 任务提供了特征融合的新思路，提示我们可以在中间层而非仅输出层进行信息整合；
分层网格搜索的高效性启发我们在超参数优化中利用问题结构设计结构化搜索策略；
EMORL 的模块化设计为动态任务场景（如对话系统）提供了灵感，是否可以根据上下文动态调整目标权重或引入新目标；
隐藏状态聚合的时间消耗问题提示未来可探索并行化聚合或更高效的生成机制以优化推理速度。