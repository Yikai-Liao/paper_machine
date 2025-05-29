---
title: "Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective"
pubDatetime: 2025-05-26T10:52:17+00:00
slug: "2025-05-trajectory-reasoning-optimization"
type: "arxiv"
id: "2505.19815"
score: 0.7925181396214541
author: "grok-3-latest"
authors: ["Junnan Liu", "Hongwei Liu", "Linchen Xiao", "Shudong Liu", "Taolin Zhang", "Zihan Ma", "Songyang Zhang", "Kai Chen"]
tags: ["LLM", "Reasoning", "Meta-Learning", "Optimization", "Trajectory"]
institution: ["Shanghai AI Laboratory"]
description: "本文提出 RaML 框架，从元学习视角将 LLM 推理轨迹建模为伪梯度更新，通过理论分析和实验验证揭示其优化本质，并提供改进推理能力的实用见解。"
---

> **Summary:** 本文提出 RaML 框架，从元学习视角将 LLM 推理轨迹建模为伪梯度更新，通过理论分析和实验验证揭示其优化本质，并提供改进推理能力的实用见解。 

> **Keywords:** LLM, Reasoning, Meta-Learning, Optimization, Trajectory

**Authors:** Junnan Liu, Hongwei Liu, Linchen Xiao, Shudong Liu, Taolin Zhang, Zihan Ma, Songyang Zhang, Kai Chen

**Institution(s):** Shanghai AI Laboratory


## Problem Background

大型语言模型（LLM）在复杂推理任务中通过链式思维（Chain-of-Thought, CoT）生成的推理轨迹显著提升了性能，但其内部推理机制仍不透明，限制了进一步的提升和泛化能力。
论文旨在从优化和元学习视角理解推理轨迹的作用，探索如何通过理论框架和训练策略改进 LLM 的推理能力。

## Method

*   **核心框架 RaML（Reasoning as Meta-Learning）**：将 LLM 的推理过程建模为元学习问题，其中每个问题被视为一个独立任务，推理轨迹被概念化为对模型参数的伪梯度更新（Pseudo-Gradient Update），类似于元学习中的内循环优化，而最终答案生成则对应外循环优化，目标是优化模型参数以适应新任务。
*   **理论基础**：基于 Transformer 架构，推理轨迹中的每个 Token 被视为对模型参数的一次伪梯度更新，通过注意力机制和前馈网络的数学推导，证明了这种更新与优化过程的相似性，借鉴了 Model-Agnostic Meta-Learning (MAML) 和 Learn-to-Optimize (L2O) 的思想。
*   **训练技术分析**：从元学习视角分析监督微调（SFT）和强化学习（RL）两种训练方法，SFT 依赖离线数据（Off-Policy）提供稳定的优化轨迹，RL 通过在线探索（On-Policy）生成推理轨迹，具有更高自由度但稳定性较低。
*   **改进策略**：通过调整推理轨迹数量（对应支持集大小）、长度（对应内循环步数）以及轨迹总结（优化路径压缩）等方法，探索推理性能和效率的提升。

## Experiment

*   **设置全面性**：实验基于数学推理任务，使用 Qwen2.5-7B-Base 模型从头训练，结合 SFT 和 Zero-GRPO（一种 RL 方法），在 AIME24、MATH500-L5 和 LiveMathBench-Hard 等数据集上评估性能，并测试跨领域泛化能力（GPQA 和 LiveCodeBench）。
*   **有效性**：SFT 在内循环优化中更稳定，性能优于 RL，但 RL 具有更高理论上限；结合 SFT 和 RL 的方法显著提升性能（如 AIME24 上 Pass@8 从 27.37% 提升到 35.87%，增幅约 31%）；更长的推理轨迹和更多轨迹数量显著提升性能和稳定性；通过总结推理轨迹减少 Token 数量，性能保持相当的同时降低了计算成本。
*   **合理性与局限**：实验覆盖多种训练策略和评估维度，指标（如 Pass@K 和 mG-Pass@K）合理，结果可信；但主要聚焦数学推理，跨领域验证不足，伪梯度更新理论基于简化假设，可能与复杂模型存在差距。

## Further Thoughts

将推理轨迹视为伪梯度更新这一视角启发我们引入优化理论（如动量、学习率调度）改进 LLM 推理效率；元学习与 LLM 推理结合提示可以借鉴任务采样和支持集设计优化训练；推理轨迹总结实验表明智能压缩策略的潜力；Token 贡献差异启发设计针对性策略增强关键 Token 作用或过滤冗余 Token。