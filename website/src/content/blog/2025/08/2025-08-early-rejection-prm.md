---
title: "Accelerating LLM Reasoning via Early Rejection with Partial Reward Modeling"
pubDatetime: 2025-08-04T00:58:56+00:00
slug: "2025-08-early-rejection-prm"
type: "arxiv"
id: "2508.01969"
score: 0.7132832827261788
author: "grok-3-latest"
authors: ["Seyyed Saeid Cheshmi", "Azal Ahmad Khan", "Xinran Wang", "Zirui Liu", "Ali Anwar"]
tags: ["LLM", "Process Reward", "Early Rejection", "Beam Search", "Reasoning"]
institution: ["University of Minnesota"]
description: "本文提出通过过程奖励模型（PRMs）作为部分奖励模型，在推理中途计算部分奖励分数以提前剔除低质量路径，显著降低计算成本（FLOPs 减少 1.4-9 倍）而不影响大型语言模型的推理性能。"
---

> **Summary:** 本文提出通过过程奖励模型（PRMs）作为部分奖励模型，在推理中途计算部分奖励分数以提前剔除低质量路径，显著降低计算成本（FLOPs 减少 1.4-9 倍）而不影响大型语言模型的推理性能。 

> **Keywords:** LLM, Process Reward, Early Rejection, Beam Search, Reasoning

**Authors:** Seyyed Saeid Cheshmi, Azal Ahmad Khan, Xinran Wang, Zirui Liu, Ali Anwar

**Institution(s):** University of Minnesota


## Problem Background

大型语言模型（LLMs）在复杂推理任务（如数学、逻辑和多步问答）中表现出色，但通过过程奖励模型（Process Reward Models, PRMs）在推理阶段评估中间步骤以提升性能会带来显著的计算开销，尤其是在并行生成大量候选解决方案时。
论文旨在解决这一问题：如何在不牺牲最终任务性能的前提下，减少推理过程中的计算资源浪费。

## Method

*   **核心思想:** 提出过程奖励模型（PRMs）同时作为部分奖励模型（Partial Reward Models）的假设，即在推理步骤部分完成时（生成少量 token 后）计算的部分奖励分数能够预测最终步骤质量，从而提前剔除低质量候选路径（Early Rejection）。
*   **具体实现:** 在束搜索（Beam Search）框架中引入提前剔除机制：
    *   在每个推理步骤生成固定数量的 token（如 τ=32 或 64）后，使用 PRM 计算部分奖励分数。
    *   根据部分奖励分数，保留高分束（top N/M beams），剔除低分束，避免对低质量路径继续生成完整步骤。
    *   对保留的束完成当前步骤生成，并继续下一轮扩展和评估。
*   **优化细节:** 采用两阶段批处理策略，初始生成阶段使用较大批次大小以提升吞吐量，完成阶段使用较小批次大小以节省内存。
*   **理论支持:** 证明在温和噪声假设下，错误剔除最优束的概率随部分生成长度（τ）增加呈指数下降，确保提前剔除的可靠性。

## Experiment

*   **有效性:** 在多个数学推理基准数据集（如 MATH-500、SAT-MATH、AIME 2024）上，提前剔除策略在不损失最终任务准确率的情况下，将推理计算量（FLOPs）减少 1.4 倍至 9 倍，尤其在 Qwen-2.5-3B 搭配 Skywork-PRM-1.5B 时效果最显著。
*   **相关性验证:** 部分奖励与最终奖励的相关性随 τ 增加而增强，在 τ=32 时相关系数达 0.78，在 τ=64 时超过 0.9，表明提前决策可靠性高。
*   **实验设置合理性:** 实验覆盖了不同规模的 LLM（如 Llama-3.2-3B、Qwen-2.5-3B）和 PRM（如 MathShepherd-7B、Skywork-1.5B），不同束宽度（4-64）和不同 τ 值（32、64、128），结果一致性强。
*   **局限性:** 实验主要聚焦数学推理任务，未涉及非单调奖励或多模态任务，内存开销和动态 τ 调整未深入探讨。

## Further Thoughts

论文提出的部分奖励预测最终质量的思路启发了我思考是否可以根据任务特性或模型生成行为动态调整提前剔除阈值（τ），以进一步优化效率；此外，是否可以将这一机制与其他搜索策略（如蒙特卡洛树搜索 MCTS）结合，或在多模态任务中通过部分生成内容（如文本+图像片段）提前判断质量，扩展其应用范围。