---
title: "Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training"
pubDatetime: 2025-09-03T15:28:51+00:00
slug: "2025-09-process-consistency-filter"
type: "arxiv"
id: "2509.03403"
score: 0.6233434522320689
author: "grok-3-latest"
authors: ["Chenlu Ye", "Zhou Yu", "Ziji Zhang", "Hao Chen", "Narayanan Sadagopan", "Jing Huang", "Tong Zhang", "Anurag Beniwal"]
tags: ["LLM", "Reinforcement Learning", "Reward Model", "Reasoning", "Data Filtering"]
institution: ["Amazon", "University of Illinois Urbana-Champaign"]
description: "本文提出过程一致性过滤（PROF）框架，通过筛选过程奖励和结果奖励一致的训练样本，显著提升数学推理任务中模型的最终准确性和推理过程质量，同时避免奖励欺骗问题。"
---

> **Summary:** 本文提出过程一致性过滤（PROF）框架，通过筛选过程奖励和结果奖励一致的训练样本，显著提升数学推理任务中模型的最终准确性和推理过程质量，同时避免奖励欺骗问题。 

> **Keywords:** LLM, Reinforcement Learning, Reward Model, Reasoning, Data Filtering

**Authors:** Chenlu Ye, Zhou Yu, Ziji Zhang, Hao Chen, Narayanan Sadagopan, Jing Huang, Tong Zhang, Anurag Beniwal

**Institution(s):** Amazon, University of Illinois Urbana-Champaign


## Problem Background

在数学推理任务中，强化学习中使用的结果奖励模型（Outcome Reward Models, ORMs）过于粗粒度，无法区分正确答案中的错误推理或错误答案中的有效推理步骤，导致训练梯度噪声和误导，限制了推理过程质量的提升。
虽然过程奖励模型（Process Reward Models, PRMs）提供了细粒度的中间步骤指导，但其不准确性和奖励欺骗（reward hacking）问题阻碍了应用。
论文旨在解决如何协调准确但粗粒度的 ORMs 和细粒度但噪声较大的 PRMs，以同时提升最终准确性和推理过程质量。

## Method

*   **核心思想:** 提出过程一致性过滤（Process Consistency Filter, PROF）框架，通过过程奖励和结果奖励的一致性筛选训练数据，避免直接融合 PRM 和 ORM 带来的奖励欺骗问题，同时提升模型性能。
*   **具体实现:** 
    *   在训练时生成多于所需数量的响应样本（rollouts），并通过结果奖励（ORM）将其分为正确和错误两组。
    *   使用预训练的过程奖励模型（PRM）为每个响应的中间步骤计算奖励，并通过平均值（加入步长正则化）得到轨迹级一致性分数。
    *   对正确和错误响应分别按一致性分数排名，剔除不一致样本（如正确答案但推理有误，或错误答案但推理部分合理），并通过计算移除数量维持正负样本平衡。
    *   最终保留的样本用于政策更新（如结合 Group Relative Policy Optimization, GRPO 算法），但 PRM 不直接参与梯度计算，仅用于筛选。
*   **关键优势:** 避免了 PRM 的噪声直接影响训练过程，通过一致性过滤减少了冲突梯度，同时模块化设计使其可与多种强化学习算法结合。

## Experiment

*   **有效性:** PROF-GRPO 在多个数学推理基准数据集（如 Math500, Minerva Math, Olympiad Bench）上显著提升了最终准确性，例如在 Qwen2.5-Math-7B-base 模型上，平均准确率从 GRPO 的 49.9% 和 Blend 方法的 47.3% 提升至 51.7%，增幅超过 4%。
*   **推理质量提升:** 中间推理步骤质量通过 Monte Carlo 估计等指标验证有显著改进，例如在 Math500 上提升 9.2%，在 Minerva Math 上提升 37.4%，远超最终准确性提升幅度。
*   **稳定性:** 学习动态显示 PROF-GRPO 收敛更快，熵损失和响应长度控制稳定，相比 Blend 方法避免了奖励欺骗和熵崩溃问题。
*   **实验全面性:** 实验覆盖 Qwen2.5-Math 和 LLaMA 不同规模模型，多个基准数据集，并通过消融研究验证了分组过滤、样本数量和过滤方式的影响，证明了方法的鲁棒性和泛化能力。

## Further Thoughts

过程一致性过滤的概念非常具有启发性，通过多维度奖励信号的一致性筛选数据，而不是简单加权融合，可以有效减少噪声影响，这种思路可推广至代码生成或多模态任务中，通过设计多层次奖励一致性提升训练质量；此外，PROF 的模块化设计表明数据筛选策略可以在不修改核心算法的情况下显著提升性能，为强化学习研究提供了一个低成本、高效的优化方向。