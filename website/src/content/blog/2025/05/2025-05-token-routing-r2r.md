---
title: "R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing"
pubDatetime: 2025-05-27T16:57:20+00:00
slug: "2025-05-token-routing-r2r"
type: "arxiv"
id: "2505.21600"
score: 0.6645604381678965
author: "grok-3-latest"
authors: ["Tianyu Fu", "Yi Ge", "Yichen You", "Enshu Liu", "Zhihang Yuan", "Guohao Dai", "Shengen Yan", "Huazhong Yang", "Yu Wang"]
tags: ["LLM", "SLM", "Token Routing", "Reasoning", "Test Time Scaling"]
institution: ["Tsinghua University", "Infinigence AI", "Shanghai Jiao Tong University"]
description: "本文提出 R2R，一种 token 级路由方法，通过动态识别并修正推理路径中的分歧 token，使小型语言模型在效率提升的同时接近大型语言模型的性能，显著推进了测试时扩展的 Pareto 前沿。"
---

> **Summary:** 本文提出 R2R，一种 token 级路由方法，通过动态识别并修正推理路径中的分歧 token，使小型语言模型在效率提升的同时接近大型语言模型的性能，显著推进了测试时扩展的 Pareto 前沿。 

> **Keywords:** LLM, SLM, Token Routing, Reasoning, Test Time Scaling

**Authors:** Tianyu Fu, Yi Ge, Yichen You, Enshu Liu, Zhihang Yuan, Guohao Dai, Shengen Yan, Huazhong Yang, Yu Wang

**Institution(s):** Tsinghua University, Infinigence AI, Shanghai Jiao Tong University


## Problem Background

大型语言模型（LLMs）在推理任务中表现出色，但生成推理路径（如 Chain-of-Thought, CoT）的高计算成本限制了其在测试时扩展（test-time scaling）中的应用；
相比之下，蒸馏后的小型语言模型（SLMs）效率更高，但其推理路径常与 LLMs 产生分歧，导致性能下降。
论文发现 SLMs 和 LLMs 在大部分 token 预测上是一致或差异无害（neutral），只有少部分 token 会导致推理路径的真正分歧（divergent），因此提出通过选择性使用 LLMs 解决关键分歧问题，以兼顾效率和性能。

## Method

*   **核心思想：** 提出 Roads to Rome (R2R)，一种 token 级别的路由方法，通过动态地在 SLM 和 LLM 之间切换，仅对导致推理路径分歧的 token 使用 LLM 修正，而对一致或无害差异的 token 使用 SLM 生成，以降低计算成本并保持推理质量。
*   **数据标注流程：** 设计自动化 pipeline 生成路由标签：
    *   首先使用 LLM 生成完整的推理路径作为参考；
    *   然后用 SLM 预填充（prefill）预测 token，识别与 LLM 预测不同的 token；
    *   对差异 token，分别从 SLM 和 LLM 的预测开始，由 LLM 继续生成后续内容（continuation），直至句子结束或特定停止条件；
    *   使用另一个 LLM 作为验证器，判断差异是‘neutral’（无害，如表达方式差异）还是‘divergent’（导致推理路径分歧），从而为每个 token 标注路由偏好（SLM 或 LLM）。
*   **神经路由器设计：** 构建一个轻量级的前馈神经网络（FFN，56M 参数）作为路由器：
    *   输入为 SLM 的输出特征，包括 top-100 logits（反映预测不确定性，如熵）、token 嵌入（反映 token 稀有性）和最后一层隐藏状态（提供语义上下文）；
    *   输出为当前 token 是否分歧的二分类概率，若概率超过预设阈值，则调用 LLM 修正该 token；
    *   训练时使用交叉熵损失，并通过类频率反比加权解决数据不平衡问题，验证集上调整阈值以控制 LLM 使用率。
*   **推理时路由方案：** 在每个 token 生成步骤，路由器基于 SLM 输出实时预测分歧概率，若超过阈值则调用 LLM 生成当前 token，避免传统推测解码中周期性验证导致的回滚开销；通过并行预填充和 KV-Cache 更新进一步优化效率。

## Experiment

*   **有效性：** 在数学（AIME）、编码（LiveCodeBench）和问答（GPQA）基准测试上，R2R 以平均 5.6B 参数规模（基于 DeepSeek R1-1.5B 和 R1-32B 组合）达到 46% 平均准确率，接近 R1-32B 的 50%，比 R1-1.5B 的 10% 提升 4.6 倍，甚至超越 R1-14B 的 43%。
*   **效率提升：** 相比 R1-32B，R2R 实现 2.8 倍 wall-clock 加速，同时保持相似性能；相比查询级路由方法，R2R 加速 1.48-1.52 倍；相比推测解码方法（如 EAGLE2, HASS），R2R 在单批次场景下表现更优。
*   **实验设置合理性：** 实验覆盖多种任务类型，数据标注和验证集设计合理，路由阈值通过验证集调整以控制 LLM 使用率（仅 11-15% token 调用 LLM）；对比了蒸馏模型、查询级路由和推测解码等多种基线，指标包括准确率、平均激活参数和总成本，设置全面。
*   **局限性：** 实验主要基于贪婪采样，未探索其他采样策略的影响；系统级优化仍有改进空间，特别是在多批次场景下。

## Further Thoughts

R2R 的 token 级别路由机制揭示了推理路径分歧的细粒度本质，启发我思考是否可以将类似动态路由策略应用于多模态模型的推理优化，或在模型蒸馏中引入分歧感知的训练目标，以进一步提升效率；此外，其自动化数据标注 pipeline 为其他需要细粒度标签的任务提供了可借鉴的框架，例如在强化学习或多模型协作场景中识别关键决策点。