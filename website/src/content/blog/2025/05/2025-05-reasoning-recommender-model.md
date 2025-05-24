---
title: "$\text{R}^2\text{ec}$: Towards Large Recommender Models with Reasoning"
pubDatetime: 2025-05-22T17:55:43+00:00
slug: "2025-05-reasoning-recommender-model"
type: "arxiv"
id: "2505.16994"
score: 0.7118862726338331
author: "grok-3-latest"
authors: ["Runyang You", "Yongqi Li", "Xinyu Lin", "Xin Zhang", "Wenjie Wang", "Wenjie Li", "Liqiang Nie"]
tags: ["LLM", "Recommendation System", "Reasoning", "Reinforcement Learning", "Autoregressive Generation"]
institution: ["The Hong Kong Polytechnic University", "National University of Singapore", "University of Science and Technology of China", "Harbin Institute of Technology (Shenzhen)"]
description: "本文提出 R[2]ec，一个统一的大型推荐模型，通过自回归过程整合推理与推荐能力，并利用 RecPO 强化学习框架实现无标注优化，显著提升推荐性能。"
---

> **Summary:** 本文提出 R[2]ec，一个统一的大型推荐模型，通过自回归过程整合推理与推荐能力，并利用 RecPO 强化学习框架实现无标注优化，显著提升推荐性能。 

> **Keywords:** LLM, Recommendation System, Reasoning, Reinforcement Learning, Autoregressive Generation

**Authors:** Runyang You, Yongqi Li, Xinyu Lin, Xin Zhang, Wenjie Wang, Wenjie Li, Liqiang Nie

**Institution(s):** The Hong Kong Polytechnic University, National University of Singapore, University of Science and Technology of China, Harbin Institute of Technology (Shenzhen)


## Problem Background

大型语言模型（LLMs）在推荐系统中展现出强大潜力，但现有方法多将推理与推荐解耦，导致资源成本高和联合优化不足的问题。
论文旨在探索如何将推理能力内嵌到大型推荐模型中，通过统一架构提升推荐性能并降低资源开销。

## Method

*   **模型设计：R[2]ec 架构**：基于解码器架构，配备两个任务特定头部：
    *   **语言建模头部（Language-Modeling Head）**：负责生成推理 token，通过自回归过程逐步构建推理序列。
    *   **推荐头部（Recommendation Head）**：基于最终隐藏状态对候选物品打分，用于物品预测。
    这种设计通过共享隐藏状态空间，将推理与推荐紧密耦合，确保推理直接影响推荐结果。
*   **优化框架：RecPO（基于强化学习）**：针对推荐领域缺乏推理标注数据的挑战，提出以下优化步骤：
    *   **轨迹采样**：为每个用户生成多条推理-推荐轨迹（Reasoning-then-Recommend Trajectories），采用温度和 top-K 采样控制随机性和多样性。
    *   **奖励与优势估计**：设计融合奖励机制，结合离散排名奖励（如 NDCG）和连续相似性奖励（如 softmax 相似度），以更细粒度评估轨迹质量，并通过优势估计（如 GRPO 或 RLOO）降低优化方差。
    *   **训练目标**：通过联合优化目标，同时更新推理和推荐能力，采用剪切比率损失（Clipped Ratio Loss）确保优化稳定性，聚焦优势最高的轨迹进行推荐学习，同时保留其他轨迹以保证推理探索性。
*   **创新点**：R[2]ec 将推理与推荐整合到单一模型中，避免了解耦设计的资源浪费，并通过 RL 框架在无标注数据的情况下实现能力提升。

## Experiment

*   **有效性**：R[2]ec 在三个 Amazon 数据集（CDs and Vinyl, Video Games, Musical Instruments）上显著优于传统推荐模型（如 GRU4Rec, SASRec）、基于 LLM 的推荐模型（如 TIGER, BigRec）以及推理增强的推荐系统（如 LangPTune），Hit@5 和 NDCG@20 分别提升了 68.67% 和 45.21%。
*   **实验设置合理性**：实验覆盖多个领域数据集，设置全集推荐以贴近实际场景，基线选择全面，消融实验验证了推理模块和融合奖励机制的重要性。
*   **局限性**：推理生成增加了推理延迟（单批次下为 16725.54ms，相较 LangPTune 的 19030.95ms 虽有改进但仍较高），且由于资源限制仅采用参数高效微调（LoRA），可能未完全发挥潜力。

## Further Thoughts

将推理能力内嵌到任务模型中并通过共享隐藏状态实现端到端优化的思路，可推广到搜索或对话系统；融合奖励机制对其他无标注生成任务具有借鉴意义；推理长度与性能的正相关性提示未来可探索动态调整推理深度以平衡性能与效率。