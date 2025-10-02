---
title: "RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs"
pubDatetime: 2025-09-29T19:33:44+00:00
slug: "2025-09-radar-reasoning-routing"
type: "arxiv"
id: "2509.25426"
score: 0.864801152923295
author: "grok-3-latest"
authors: ["Nigel Fernandez", "Branislav Kveton", "Ryan A. Rossi", "Andrew S. Lan", "Zichao Wang"]
tags: ["LLM", "Reasoning", "Routing", "Performance Optimization", "Cost Efficiency"]
institution: ["University of Massachusetts Amherst", "Adobe Research"]
description: "本文提出 RADAR 框架，通过项目反应理论和多目标优化，为推理型语言模型的查询动态选择模型配置，实现性能与成本的最优平衡，并在多个基准上显著优于现有方法。"
---

> **Summary:** 本文提出 RADAR 框架，通过项目反应理论和多目标优化，为推理型语言模型的查询动态选择模型配置，实现性能与成本的最优平衡，并在多个基准上显著优于现有方法。 

> **Keywords:** LLM, Reasoning, Routing, Performance Optimization, Cost Efficiency

**Authors:** Nigel Fernandez, Branislav Kveton, Ryan A. Rossi, Andrew S. Lan, Zichao Wang

**Institution(s):** University of Massachusetts Amherst, Adobe Research


## Problem Background

推理型语言模型（Reasoning Language Models, RLMs）在数学、科学和编程等复杂任务上表现出色，但不同模型大小和推理预算（reasoning budget）显著影响性能和成本。
论文的出发点是解决如何为每个查询动态选择合适的模型配置（即模型大小和推理预算），以在性能和成本之间达到最优平衡，避免资源浪费或性能不足的问题。
关键问题在于，简单的查询可能仅需小型模型和低推理预算，而复杂查询则需要强大模型和更高预算，现有方法缺乏对查询难度和模型能力的动态评估。

## Method

*   **核心思想:** 提出 RADAR（Reasoning–Ability and Difficulty-Aware Routing）框架，通过路由机制为每个查询选择最优的模型配置，在性能和成本之间实现动态平衡。
*   **配置离散化:** 将问题统一为模型和推理预算的组合（model-budget pairs），将路由问题转化为多目标优化（Multi-Objective Optimization, MOO），目标是最大化性能、最小化成本。
*   **项目反应理论（IRT）建模:** 借鉴心理测量学中的两参数逻辑模型（2PL IRT），通过查询响应数据联合估计查询难度（difficulty）和模型配置能力（ability），并利用查询嵌入（通过冻结的嵌入模型生成）增强对未见查询的泛化能力，参数具有可解释性。
*   **多目标优化求解:** 采用线性标量化（Linear Scalarization）和切比雪夫标量化（Chebyshev Scalarization）两种方法，将多目标问题转化为单目标问题，探索性能-成本的 Pareto 前沿，根据用户指定的权衡权重选择配置。
*   **成本预测:** 使用基于启发式的成本预测方法，结合每个配置的每 token 成本和生成的推理及完成 token 数量，计算查询成本。
*   **自适应测试扩展:** 对于新加入的模型配置，通过动态选择少量评估查询（基于 Fisher 信息优先级），快速估计其能力，实现框架的可扩展性。
*   **黑箱设置与低延迟:** RADAR 不需访问模型权重，仅通过 API 调用工作，路由决策在查询前完成（延迟约 7 毫秒），避免生成过程中模型切换或 KV 缓存重计算。

## Experiment

*   **有效性:** RADAR 在 8 个推理基准（如 MATH-500, GPQA-Diamond, FRAMES）上显著优于现有路由方法（如 RouterBench, IRT-Router），例如在 GPQA-Diamond 上 hypervolume 指标提升 8%，在 MATH-500 上以 1.31% 成本实现 OpenAI o4-mini 高推理预算 90% 的性能。
*   **泛化能力:** 在分布外（OOD）查询上表现优异，尤其在长上下文多文档问答任务（FRAMES）上，尽管训练数据以短查询为主，表明 IRT 模型和查询嵌入设计有效捕捉查询难度。
*   **可扩展性:** 通过自适应测试，仅用 12% 训练查询即可准确估计新模型（如 Qwen3-14B）能力，添加新配置后路由性能进一步提升。
*   **实验设置合理性:** 实验覆盖 35 种模型配置（包括 OpenAI o4-mini 和 Qwen3 系列），数据集多样（数学、科学、法律等领域），评估了分布内（ID）和分布外（OOD）场景，指标（如 hypervolume, CPT）多维度衡量性能-成本权衡，设置全面。
*   **局限性:** 在某些高难度 OOD 查询（如 AIME）上，RADAR 倾向分配能力稍低配置，导致性能略降，可能与训练数据中高难度样本不足有关。

## Further Thoughts

RADAR 的 IRT 应用为模型路由提供了可解释性，未来可探索将其扩展到多模态任务，评估图像或语音输入的难度；MOO 框架为引入更多优化维度（如延迟、碳排放）提供了可能性；此外，可结合强化学习通过用户反馈或任务结果实时优化路由策略，而非依赖预设权重。