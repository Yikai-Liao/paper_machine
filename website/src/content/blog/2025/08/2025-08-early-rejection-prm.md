---
title: "Accelerating LLM Reasoning via Early Rejection with Partial Reward Modeling"
pubDatetime: 2025-08-04T00:58:56+00:00
slug: "2025-08-early-rejection-prm"
type: "arxiv"
id: "2508.01969"
score: 0.7132832827261788
author: "grok-3-latest"
authors: ["Seyyed Saeid Cheshmi", "Azal Ahmad Khan", "Xinran Wang", "Zirui Liu", "Ali Anwar"]
tags: ["LLM", "Reasoning", "Process Reward", "Early Rejection", "Compute Efficiency"]
institution: ["University of Minnesota"]
description: "本文提出并验证了过程奖励模型（PRM）作为部分奖励模型的假设，通过 Early Rejection 机制在推理中提前剔除低质量路径，显著降低大型语言模型的计算开销达 1.4 倍至 9 倍，同时保持任务性能。"
---

> **Summary:** 本文提出并验证了过程奖励模型（PRM）作为部分奖励模型的假设，通过 Early Rejection 机制在推理中提前剔除低质量路径，显著降低大型语言模型的计算开销达 1.4 倍至 9 倍，同时保持任务性能。 

> **Keywords:** LLM, Reasoning, Process Reward, Early Rejection, Compute Efficiency

**Authors:** Seyyed Saeid Cheshmi, Azal Ahmad Khan, Xinran Wang, Zirui Liu, Ali Anwar

**Institution(s):** University of Minnesota


## Problem Background

大型语言模型（LLMs）在数学、逻辑和多步推理任务中表现出强大的能力，但通过扩展推理时计算（如生成多个候选路径并使用过程奖励模型 PRM 评估）来提升性能的方法带来了巨大的计算开销。
论文聚焦于解决这一问题：如何在不牺牲推理质量的前提下减少计算资源浪费，尤其是在传统 PRM 方法中需要完整生成每个推理步骤后才进行评估，导致大量低质量路径占用计算资源。

## Method

*   **核心假设:** 过程奖励模型（PRM）可以作为部分奖励模型（Partial Reward Model），即在推理步骤部分完成时，PRM 给出的部分奖励分数与最终完整步骤的分数高度相关，可用于提前判断候选路径质量。
*   **具体流程:** 在束搜索（Beam Search）框架中引入 Early Rejection 机制：
    1. 在每个推理步骤中，先生成一小部分 token（定义为 τ 个 token，如 32 或 64 个）。
    2. 使用 PRM 对部分生成的序列计算部分奖励分数（Partial Reward）。
    3. 根据部分奖励分数筛选出得分较高的束（top N/M），仅对这些束继续生成剩余 token 完成当前步骤，剔除低分束以节省计算。
    4. 对筛选出的束进行扩展，生成下一轮候选路径，重复上述过程直到满足停止条件。
*   **理论支持:** 论文通过理论分析证明，部分奖励与最终奖励的相关性随生成长度（τ）增加而增强，且误剔最优束的风险随生成长度呈指数下降。
*   **优势:** 该方法无需额外训练模型，仅通过调整推理流程即可实现计算优化，且复用现有 PRM，易于集成到现有系统中。

## Experiment

*   **计算效率提升:** 在 MATH-500、SAT-MATH 和 AIME 2024 等数学推理基准数据集上，Early Rejection 方法显著降低了推理时的计算开销（以 FLOPs 计），实现了 1.4 倍到 9 倍的计算量减少，尤其在 τ=64 时效果最佳。
*   **准确率表现:** 在大多数配置下，该方法保持了与传统束搜索（Vanilla Beam Search）相当的准确率，甚至在某些情况下（如 Qwen-2.5B 在小束宽度时）略有提升，表明提前剔除低质量路径并未显著误剔最优路径。
*   **实验设置合理性:** 实验覆盖了两种 LLM（Llama-3.2-3B 和 Qwen-2.5-3B）和两种 PRM（MathShepherd-Mistral-7B 和 Skywork-PRM-1.5B），测试了多种 τ 值（32、64、128）和束宽度（4 到 64），设置全面且合理。
*   **额外观察:** 较小的 PRM（如 Skywork-1.5B）在某些情况下表现优于较大 PRM，表明 PRM 规模并非决定性因素；同时，方法对生成行为更探索性的模型（如 Qwen）效果更显著。

## Further Thoughts

论文提出的部分奖励预测最终质量的假设可能不仅限于数学推理，若能在其他领域（如代码生成或多模态任务）验证其通用性，Early Rejection 或将成为一种广泛适用的计算优化策略。
此外，是否可以根据任务特性动态调整 τ 值，例如在推理早期使用较小 τ 快速筛选，后期使用较大 τ 提高精度？
另外，Early Rejection 是否可与强化学习（如 RLHF）结合，通过学习最优剔除策略进一步提升效率？