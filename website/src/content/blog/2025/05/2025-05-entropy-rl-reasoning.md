---
title: "The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models"
pubDatetime: 2025-05-28T17:38:45+00:00
slug: "2025-05-entropy-rl-reasoning"
type: "arxiv"
id: "2505.22617"
score: 0.7100558833630352
author: "grok-3-latest"
authors: ["Ganqu Cui", "Yuchen Zhang", "Jiacheng Chen", "Lifan Yuan", "Zhi Wang", "Yuxin Zuo", "Haozhan Li", "Yuchen Fan", "Huayu Chen", "Weize Chen", "Zhiyuan Liu", "Hao Peng", "Lei Bai", "Wanli Ouyang", "Yu Cheng", "Bowen Zhou", "Ning Ding"]
tags: ["LLM", "Reinforcement Learning", "Policy Entropy", "Exploration", "Reasoning"]
institution: ["Shanghai AI Laboratory", "Tsinghua University", "UIUC", "Peking University", "Nanjing University", "CUHK"]
description: "本文揭示了强化学习中策略熵崩溃的机制，并通过`Clip-Cov`和`KL-Cov`方法干预高协方差token，维持探索能力，显著提升大型语言模型推理性能。"
---

> **Summary:** 本文揭示了强化学习中策略熵崩溃的机制，并通过`Clip-Cov`和`KL-Cov`方法干预高协方差token，维持探索能力，显著提升大型语言模型推理性能。 

> **Keywords:** LLM, Reinforcement Learning, Policy Entropy, Exploration, Reasoning

**Authors:** Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, Ning Ding

**Institution(s):** Shanghai AI Laboratory, Tsinghua University, UIUC, Peking University, Nanjing University, CUHK


## Problem Background

在强化学习（RL）应用于大型语言模型（LLM）推理能力提升的过程中，策略熵（Policy Entropy）的急剧下降是一个主要障碍，导致模型过早丧失探索能力，性能达到饱和。
作者通过实验发现，熵与下游性能之间存在可预测的指数关系（R = -a exp H + b），表明性能提升是以熵消耗为代价的，且性能上限是确定的，限制了RL的扩展性。

## Method

*   **核心思想:** 通过分析策略熵的动态变化，设计方法干预熵下降过程，维持探索能力以提升RL在LLM推理中的性能。
*   **理论基础:** 作者推导了熵变化与动作概率和优势值（Advantage）协方差（Covariance）的关系：高概率高优势动作降低熵，低概率高优势动作增加熵。
*   **具体实现:** 提出了两种熵控制方法：
    *   **Clip-Cov:** 计算每个token的协方差值，随机选择一小部分高协方差token（比例约为2×10^-4），将其梯度从策略更新中分离（detach），避免这些token过度影响熵下降。
    *   **KL-Cov:** 对具有最高协方差的token（比例约为2×10^-3或更低）施加KL散度惩罚，限制其更新幅度，控制熵下降，同时通过调整KL系数（β）调节干预强度。
*   **关键点:** 这两种方法仅针对少量关键token进行干预，不直接修改整体损失函数，避免了传统熵正则化（如熵损失或KL正则化）带来的不稳定性和性能下降问题。

## Experiment

*   **有效性:** 实验表明，`Clip-Cov`和`KL-Cov`方法显著提升了策略熵水平（例如`KL-Cov`在训练后期熵值比基线高10倍），并在多个数学推理和代码生成基准（如AIME24、MATH-500）上取得性能提升，Qwen2.5-7B平均提升2.0%，Qwen2.5-32B平均提升6.4%，在挑战性任务AIME24上提升高达15.0%。
*   **优越性:** 相比基线方法（如GRPO和Clip-higher），作者的方法在熵控制和性能提升上更稳定，避免了性能饱和或下降，特别是在训练后期仍能持续探索。
*   **实验设置:** 实验覆盖了多种模型规模（0.5B到32B）、任务领域（数学和代码）、RL算法（GRPO、REINFORCE++等）以及多个公开数据集，设置全面合理，增强了结果的普适性。
*   **局限性:** 熵控制的超参数（如Clip比例和KL系数）对结果敏感，最优熵值仍未明确，需进一步研究。

## Further Thoughts

熵与性能的可预测关系（R = -a exp H + b）为早期预测模型性能上限提供了可能，节省计算资源；此外，少量高协方差token对熵动态的显著影响提示LLM决策可能高度集中于关键点，未来可通过分析这些token的语义或位置特征优化熵控制策略；同时，更大模型在熵控制后性能提升更明显，是否可以通过自适应策略根据模型规模动态调整干预强度，值得探索。