---
title: "Rethinking Entropy Regularization in Large Reasoning Models"
pubDatetime: 2025-09-29T17:49:25+00:00
slug: "2025-09-selective-entropy-regularization"
type: "arxiv"
id: "2509.25133"
score: 0.742071249188738
author: "grok-3-latest"
authors: ["Yuxian Jiang", "Yafu Li", "Guanxu Chen", "Dongrui Liu", "Yu Cheng", "Jing Shao"]
tags: ["LLM", "Reasoning", "RLHF", "Sampling", "Post-Training"]
institution: ["Shanghai Artificial Intelligence Laboratory", "Fudan University", "Shanghai Jiao Tong University", "Chinese University of Hong Kong"]
description: "本文提出 SIREN 方法，通过选择性熵正则化和自锚定机制解决大型推理模型在 RLVR 中的熵崩溃和过早收敛问题，显著提升推理性能和探索能力。"
---

> **Summary:** 本文提出 SIREN 方法，通过选择性熵正则化和自锚定机制解决大型推理模型在 RLVR 中的熵崩溃和过早收敛问题，显著提升推理性能和探索能力。 

> **Keywords:** LLM, Reasoning, RLHF, Sampling, Post-Training

**Authors:** Yuxian Jiang, Yafu Li, Guanxu Chen, Dongrui Liu, Yu Cheng, Jing Shao

**Institution(s):** Shanghai Artificial Intelligence Laboratory, Fudan University, Shanghai Jiao Tong University, Chinese University of Hong Kong


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）在通过强化学习与可验证奖励（RLVR）提升推理能力时，面临熵崩溃（entropy collapse）和过早收敛（premature convergence）的问题，导致模型输出响应单一，探索能力受限，影响训练效率和性能。
传统的熵正则化方法由于 LRMs 巨大的动作空间和长序列特性，容易引发全局熵爆炸（global entropy explosion），无法有效解决这一问题。

## Method

*   **核心思想:** 提出 SIREN（Selective Entropy Regularization），通过限制熵正则化到有意义的动作和状态子集，实现更有针对性的探索，避免熵崩溃和过早收敛。
*   **具体实现:** SIREN 包含以下关键组件：
    *   **Top-p Mask:** 在每个 token 的概率分布中，仅对排名靠前的语义上有意义的 token 子集（称为 policy nucleus）计算熵，限制探索范围，避免在整个词汇表上无差别探索导致的熵爆炸。
    *   **Peak-Entropy Mask:** 在长序列中，通过熵值分位数识别关键 token（如逻辑连接词），仅对这些 token 应用熵正则化，防止熵在整个序列上累积爆炸。
    *   **Self-Anchored Regularization:** 将熵正则化目标从最大化熵改为与初始熵值（预训练模型的熵）保持接近，使用均方误差（MSE）损失来稳定训练，避免熵过高或过低。
*   **技术基础:** SIREN 基于 Dr.GRPO（一种改进的强化学习算法），通过上述机制在不引入过多计算开销的情况下优化探索与性能的平衡。
*   **关键优势:** 不需要修改模型架构，仅通过正则化策略调整即可实现显著改进，同时对超参数的敏感性较低。

## Experiment

*   **有效性:** 在 Qwen2.5-Math-7B 模型上，SIREN 显著提升了性能，平均 maj@k 达到 54.6，较最强基线提升 +4.8，avg@k 提升 +1.6，尤其在困难数据集 AIME24/25 上提升达 +6.6。
*   **普适性:** 在较小模型 Qwen2.5-Math-1.5B 和较弱模型 LLaMa3.1-8B 上，SIREN 同样表现出色，平均 maj@k 分别提升 +2.4 和 +2.8，显示出方法对不同规模和能力模型的适用性。
*   **实验设置合理性:** 实验覆盖多个数学推理基准数据集（AIME24/25, MATH500 等），评估指标包括 maj@k, avg@k 和 pass@k，全面反映模型推理能力和探索多样性；同时通过 perplexity 和熵动态分析验证了 SIREN 在维持响应多样性和适当熵水平方面的优势。
*   **训练动态:** SIREN 在训练早期维持较高熵以鼓励探索，后期逐渐收敛，避免了基线方法中的熵爆炸或过早下降问题，训练更稳定。
*   **消融分析:** 消融实验表明 SIREN 的三个组件均有贡献，其中自锚定正则化对性能影响最大，去除后 maj@k 和 avg@k 分别下降 10.3 和 15.5。
*   **计算开销:** SIREN 的额外开销主要来自掩码计算和熵调整，相对较小，未显著增加训练负担。

## Further Thoughts

SIREN 的选择性探索理念启发我们可以在其他任务（如代码生成或多模态推理）中设计任务特定的关键子集选择机制；自锚定正则化的思想可扩展到其他超参数敏感问题，通过初始状态作为动态参考点优化训练目标；此外，是否可以结合正确性反馈或重要性采样等信号进一步提升熵掩码的精准性？