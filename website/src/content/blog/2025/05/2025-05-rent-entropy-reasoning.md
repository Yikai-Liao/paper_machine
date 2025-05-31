---
title: "Maximizing Confidence Alone Improves Reasoning"
pubDatetime: 2025-05-28T17:59:37+00:00
slug: "2025-05-rent-entropy-reasoning"
type: "arxiv"
id: "2505.22660"
score: 0.7556104363201532
author: "grok-3-latest"
authors: ["Mihir Prabhudesai", "Lili Chen", "Alex Ippoliti", "Katerina Fragkiadaki", "Hao Liu", "Deepak Pathak"]
tags: ["LLM", "Reinforcement Learning", "Entropy Minimization", "Reasoning", "Unsupervised Learning"]
institution: ["Carnegie Mellon University"]
description: "本文提出 RENT 方法，通过无监督强化学习利用模型预测分布的负熵作为奖励信号，显著提升大型语言模型在推理任务中的性能。"
---

> **Summary:** 本文提出 RENT 方法，通过无监督强化学习利用模型预测分布的负熵作为奖励信号，显著提升大型语言模型在推理任务中的性能。 

> **Keywords:** LLM, Reinforcement Learning, Entropy Minimization, Reasoning, Unsupervised Learning

**Authors:** Mihir Prabhudesai, Lili Chen, Alex Ippoliti, Katerina Fragkiadaki, Hao Liu, Deepak Pathak

**Institution(s):** Carnegie Mellon University


## Problem Background

大型语言模型（LLM）在推理任务（如数学、科学问题求解）中表现出色，但传统强化学习方法依赖外部监督（如基于正确答案的奖励），在现实世界或开放性场景中往往缺乏这种监督。
论文旨在解决如何在无外部反馈的情况下，利用模型自身的内在信号（如信心或不确定性）来提升推理能力，特别是在标注数据不可用时改进模型性能。

## Method

*   **核心思想:** 提出 RENT（Reinforcement Learning via Entropy Minimization），一种完全无监督的强化学习方法，利用模型预测分布的熵（entropy）作为内在奖励信号，鼓励模型生成更自信（低熵）的输出以提升推理能力。
*   **奖励设计:** 将奖励定义为负熵（negative entropy），即模型在生成响应时，计算每个 token 概率分布的熵，并取整个响应的平均负熵作为奖励，熵越低表示模型越自信，奖励越高。
*   **优化策略:** 采用 Group Relative Policy Optimization (GRPO) 算法，通过与一组基准策略的相对性能比较来优化当前策略，增强学习稳定性，尤其在无监督奖励信号可能噪声较大的情况下。
*   **重点关注:** 通过实验分析发现，响应中靠近最终答案的 token（尤其是最后部分）的熵与准确性相关性更高，因此优先优化这些 token 的信心，而非均匀对待所有 token。
*   **实现细节:** 不需要修改模型结构，仅通过调整奖励函数和优化过程即可实现，适用于不同任务和模型规模，且奖励信号密集、通用且易于计算。

## Experiment

*   **有效性:** 在多个推理基准数据集（GSM8K, MATH500, AMC, AIME, GPQA）上，RENT 方法显著提升了模型性能，例如 Qwen2.5-7B-Instruct 在 GSM8K 上从 0.780 提升到 0.900，Qwen2.5-Math-7B 从 0.000 提升到 0.645，显示出明显的改进。
*   **对比分析:** 与仅使用格式奖励（format reward）的基线相比，RENT 通常表现更好，表明其提升不仅仅来自格式学习；与另一无监督方法 TTRL（基于多数投票奖励）相比，RENT 在困难任务（如 AIME）上表现更优，准确率从 0.172 提升到 0.270。
*   **实验设置:** 实验覆盖了多种数据集（数学、科学领域）和模型（Qwen 和 Mistral 系列，参数规模从 1.5B 到 7B），设置较为全面合理；通过定性分析验证了模型学会了有意义的推理步骤。
*   **局限性:** 无监督方法可能导致过自信（overconfidence）问题，模型可能自信地给出错误答案，且无法完全媲美有监督方法，但整体上信心与准确性相关性较高，性能提升稳定。

## Further Thoughts

RENT 方法利用熵作为内在奖励信号的思路启发我们，是否可以探索其他模型内部信号（如 token 间的语义一致性或推理步骤的逻辑连贯性）来设计无监督学习方法？此外，论文发现最终答案附近的 token 信心与准确性相关性更高，这是否意味着可以动态调整奖励机制，逐步增加对后期推理步骤的重视？另外，这种基于不确定性分布的优化思路是否可以扩展到多模态任务（如图像生成），通过最小化生成过程中的不确定性来提升输出质量？