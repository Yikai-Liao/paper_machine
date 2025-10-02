---
title: "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning"
pubDatetime: 2025-09-29T07:19:34+00:00
slug: "2025-09-evolution-strategies-llm-tuning"
type: "arxiv"
id: "2509.24372"
score: 0.5483419210866527
author: "grok-3-latest"
authors: ["Xin Qiu", "Yulu Gan", "Conor F. Hayes", "Qiyao Liang", "Elliot Meyerson", "Babak Hodjat", "Risto Miikkulainen"]
tags: ["LLM", "Fine-Tuning", "Evolution Strategies", "Reinforcement Learning", "Parameter Space"]
institution: ["Cognizant AI Lab", "MIT", "UT Austin"]
description: "本文首次将进化策略（ES）扩展到大型语言模型的全参数空间微调，证明其在样本效率、长距离奖励处理、模型鲁棒性和稳定性上优于强化学习方法，为 LLMs 微调开辟了新方向。"
---

> **Summary:** 本文首次将进化策略（ES）扩展到大型语言模型的全参数空间微调，证明其在样本效率、长距离奖励处理、模型鲁棒性和稳定性上优于强化学习方法，为 LLMs 微调开辟了新方向。 

> **Keywords:** LLM, Fine-Tuning, Evolution Strategies, Reinforcement Learning, Parameter Space

**Authors:** Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Elliot Meyerson, Babak Hodjat, Risto Miikkulainen

**Institution(s):** Cognizant AI Lab, MIT, UT Austin


## Problem Background

大型语言模型（LLMs）在下游任务中的微调是 AI 部署的关键步骤，但当前主流的强化学习（RL）方法在样本效率低、处理长距离奖励困难、对基础模型敏感、易发生奖励作弊（reward hacking）以及训练不稳定等方面存在显著局限。
进化策略（ES）作为一种基于种群的零阶优化方法，过去在小规模模型上显示出与 RL 相当的性能，但由于对大规模模型可扩展性的悲观认知而被忽视。
本文旨在挑战这一认知，探索 ES 是否能直接在 LLMs 的全参数空间（数十亿参数）上进行高效微调，并克服 RL 的上述问题。

## Method

*   **核心思想:** 提出一种简化的进化策略（ES）变体，直接在 LLMs 的全参数空间中进行搜索和优化，而非 RL 常见的动作空间探索，以实现高效微调。
*   **算法框架:** 基于自然进化策略（NES）和 OpenAI ES 的思想，在每轮迭代中，通过对模型参数添加高斯噪声生成一组扰动模型（种群），评估每个扰动模型在目标任务上的奖励，然后根据归一化后的奖励加权聚合扰动来更新模型参数。
*   **实施优化:** 
    *   **内存效率:** 通过存储随机种子而非噪声本身，结合层级原地扰动与恢复，避免存储多个模型副本，显著降低 GPU 内存需求。
    *   **并行计算:** 利用多进程并行评估扰动模型，提升计算效率。
    *   **奖励归一化:** 使用 z-score 归一化奖励，确保跨迭代和任务的奖励尺度一致。
    *   **贪婪解码:** 在评估时采用贪婪解码，确保性能差异来源于参数空间探索而非动作空间随机性。
    *   **分解更新:** 参数更新按层和种子逐步进行，进一步减少峰值内存占用。
*   **关键特性:** 不需要反向传播，仅依赖前向推理，适合大规模模型；通过参数空间噪声注入，降低生成序列的方差，提供更稳定的优化路径。

## Experiment

*   **有效性:** 在 Countdown 任务（符号推理基准）上，ES 在 Qwen2.5 和 LLaMA3 系列模型（0.5B 到 8B 参数规模）上显著优于 RL 方法（PPO 和 GRPO），平均准确率提升 36.4%，而 PPO 和 GRPO 仅为 17.9% 和 21.3%-21.4%；在简洁性任务中，ES 在奖励和 KL 散度的 Pareto 前沿上优于 GRPO。
*   **样本效率:** 尽管在数十亿参数空间搜索，ES 比 RL 更高效，仅需 RL 约 20% 的样本评估即可达到同等性能。
*   **鲁棒性:** ES 对不同规模和家族的模型表现出一致的改进效果，尤其在小模型（如 Qwen2.5-0.5B）上有效，而 RL 几乎无改进；ES 还无需 KL 惩罚即可避免奖励作弊，行为更稳定。
*   **稳定性:** 跨多次运行的奖励和 KL 散度标准差远低于 RL，表明 ES 微调结果更可靠。
*   **实验设置合理性:** 实验覆盖多种模型规模和任务类型，RL 方法进行了超参数网格搜索以确保公平对比，而 ES 使用固定超参数，凸显其鲁棒性；任务设计考虑了长距离奖励和行为特性，评估全面。

## Further Thoughts

ES 通过参数空间噪声注入平滑‘崎岖’奖励景观的机制，启发我们设计其他平滑化策略来改进优化算法；小种群规模在高维参数空间的成功，可能与 LLMs 的低内在维度性有关，提示未来可探索模型参数的内在结构以优化微调；ES 不依赖过程奖励的特性，为基于内部行为（如语义熵）的无监督微调提供了新思路，可能在超级智能系统中发挥作用。