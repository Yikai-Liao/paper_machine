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
description: "本文揭示了强化学习中大语言模型推理任务的策略熵崩溃机制，并提出`Clip-Cov`和`KL-Cov`方法通过限制高协方差token更新维持探索能力，显著提升性能。"
---

> **Summary:** 本文揭示了强化学习中大语言模型推理任务的策略熵崩溃机制，并提出`Clip-Cov`和`KL-Cov`方法通过限制高协方差token更新维持探索能力，显著提升性能。 

> **Keywords:** LLM, Reinforcement Learning, Policy Entropy, Exploration, Reasoning

**Authors:** Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, Ning Ding

**Institution(s):** Shanghai AI Laboratory, Tsinghua University, UIUC, Peking University, Nanjing University, CUHK


## Problem Background

在强化学习（RL）应用于大语言模型（LLMs）推理任务的过程中，策略熵（Policy Entropy）在训练早期急剧下降，导致模型过早丧失探索能力，性能达到饱和，无法进一步提升。
作者通过实验发现熵与下游性能之间存在可预测的指数关系（R = -a exp H + b），表明性能提升以熵消耗为代价，且存在明确上限，这一问题对RL的扩展性构成了瓶颈。

## Method

*   **理论基础：** 分析策略熵的动态变化，发现熵变化与动作概率和优势值（Advantage）的协方差（Covariance）相关，高概率高优势动作降低熵，低概率高优势动作增加熵。
*   **具体方法：** 提出两种熵控制技术以限制高协方差token的更新影响，维持探索能力：
    *   **Clip-Cov：** 计算每个token的协方差，随机选择一小部分高协方差token（通过预定义的上下界和剪切比例），将其从策略梯度更新中分离（detach），避免这些token对熵的过度影响。
    *   **KL-Cov：** 同样计算token协方差，对排名前k比例的高协方差token施加KL散度惩罚（与旧策略比较），通过调整惩罚系数控制更新幅度，减缓熵下降。
*   **实现细节：** 方法仅干预少量关键token（比例在10^-4到10^-3），通过超参数（如剪切比例r和KL系数β）灵活调控熵水平，不直接修改模型架构，仅在更新阶段进行干预。
*   **目标：** 在不牺牲性能的前提下，通过维持较高熵水平促进探索，打破性能饱和瓶颈。

## Experiment

*   **实验设置：** 在多个模型（如Qwen2.5系列，0.5B到32B参数规模）和数学推理任务（如MATH500, AIME, AMC等）上测试，覆盖不同模型家族、任务难度和RL算法（如GRPO, REINFORCE++），设置全面合理。
*   **效果显著性：** 相比基线GRPO，`Clip-Cov`和`KL-Cov`显著提升性能，例如在Qwen2.5-32B上，`KL-Cov`在AIME24和AIME25数据集上分别提升15.0%和14.6%，在7B模型上平均提升2.0%，在32B模型上平均提升6.4%。
*   **熵控制能力：** 方法维持了更高的熵水平（如`KL-Cov`熵值比基线高10倍以上），避免了熵崩溃，同时响应长度增加，表明探索能力增强。
*   **对比分析：** 与传统熵正则化（如熵损失或参考模型KL惩罚）相比，提出的方法对超参数不敏感，训练更稳定，且不会导致性能下降。
*   **结论：** 实验数据支持方法的有效性，尤其在大模型上提升更显著，表明熵控制对释放模型潜力至关重要。

## Further Thoughts

熵与性能的指数关系提示是否可以通过动态调整熵目标值来优化训练，例如早期高熵鼓励探索，后期低熵稳定性能；
高协方差token的关键作用启发是否可以研究其语义特性，设计更精细干预；
大模型上效果更显著是否意味着RL扩展性瓶颈主要来自探索不足，未来可结合预训练和后训练阶段的熵管理实现更高效训练。