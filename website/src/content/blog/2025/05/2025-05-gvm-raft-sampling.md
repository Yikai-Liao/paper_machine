---
title: "Optimizing Chain-of-Thought Reasoners via Gradient Variance Minimization in Rejection Sampling and RL"
pubDatetime: 2025-05-05T06:26:00+00:00
slug: "2025-05-gvm-raft-sampling"
type: "arxiv"
id: "2505.02391"
score: 0.5793573560623108
author: "grok-3-latest"
authors: ["Jiarui Yao", "Yifan Hao", "Hanning Zhang", "Hanze Dong", "Wei Xiong", "Nan Jiang", "Tong Zhang"]
tags: ["LLM", "Chain of Thought", "Sampling", "Variance Minimization", "Reinforcement Learning"]
institution: ["University of Illinois Urbana-Champaign", "Salesforce AI Research"]
description: "本文提出GVM-RAFT方法，通过动态采样分配策略最小化梯度方差，显著提升大型语言模型在链式思维推理任务中的训练效率和性能。"
---

> **Summary:** 本文提出GVM-RAFT方法，通过动态采样分配策略最小化梯度方差，显著提升大型语言模型在链式思维推理任务中的训练效率和性能。 

> **Keywords:** LLM, Chain of Thought, Sampling, Variance Minimization, Reinforcement Learning

**Authors:** Jiarui Yao, Yifan Hao, Hanning Zhang, Hanze Dong, Wei Xiong, Nan Jiang, Tong Zhang

**Institution(s):** University of Illinois Urbana-Champaign, Salesforce AI Research


## Problem Background

大型语言模型（LLMs）在数学推理任务中通过链式思维（Chain-of-Thought, CoT）生成中间推理步骤以提升准确性，但传统拒绝采样微调方法（如RAFT）采用统一推理预算，无法根据提示难度和收敛行为动态分配计算资源，导致梯度估计方差高，训练效率低下。
关键问题在于如何在有限计算预算下，通过动态调整采样策略减少梯度方差，从而加速模型收敛并提升性能。

## Method

*   **核心思想:** 提出GVM-RAFT（Gradient Variance Minimization with RAFT），一种基于期望最大化（EM）框架的动态采样分配策略，通过最小化随机梯度方差来优化计算资源分配。
*   **具体实现:**
    *   在E步（Expectation）中，通过拒绝采样近似CoT推理的后验分布，并根据每个提示的接受率（反映难度）和梯度范数（反映对训练的贡献）动态计算采样预算。
    *   引入正则化项（通过超参数α和β控制），避免资源过度分配到极难提示，确保分配稳定性。
    *   在M步（Maximization）中，利用分配后的采样数据对模型参数进行梯度下降更新。
    *   提供理论分析，证明在平滑性和凸性假设下，该方法具有加速收敛的保证。
*   **扩展性:** GVM策略不仅适用于RAFT++，还被成功应用于强化学习（RL）算法如GRPO，显示出通用性。
*   **关键创新:** 通过结合任务难度和梯度信息动态调度资源，克服了传统静态采样策略的局限性，同时保持算法的在线性和策略性。

## Experiment

*   **有效性:** 在Qwen2.5-Math-1.5B模型上，GVM-RAFT++相比传统RAFT++在多个数学推理基准（如Math500, Minerva Math）上实现了2-4倍的收敛速度提升，最终准确率提升1.25%-5%；在Qwen2.5-Math-7B模型上，性能与基线相当，但收敛速度更快。
*   **全面性:** 实验覆盖了不同模型规模（1.5B和7B）、不同采样预算（N'和N分别为8、16、32等）、不同算法变体（RAFT++和GRPO），并通过消融研究验证了超参数（如α、β）和采样策略的影响。
*   **合理性:** 资源分配倾向于困难提示，但通过正则化避免了过度偏向，分配稳定；实验还显示较大采样预算对收敛率提升有限，提示在实际应用中可选择较小预算以降低成本。
*   **开销:** 主要增加了预采样阶段（估计接受率和梯度范数）的计算成本，但整体效率提升显著，特别是在资源受限场景下。

## Further Thoughts

GVM通过任务难度和梯度贡献动态分配资源的思路非常具有启发性，可以进一步推广到其他领域，如多任务学习中根据任务重要性动态调整训练资源，或在个性化推荐系统中根据用户反馈调整模型更新频率；此外，是否可以通过结合上下文信息（如训练进展或任务优先级）设计自适应采样预算，进一步提升效率和泛化能力？