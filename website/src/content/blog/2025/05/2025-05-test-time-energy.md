---
title: "The Energy Cost of Reasoning: Analyzing Energy Usage in LLMs with Test-time Compute"
pubDatetime: 2025-05-20T02:35:59+00:00
slug: "2025-05-test-time-energy"
type: "arxiv"
id: "2505.14733"
score: 0.8384883626433494
author: "grok-3-latest"
authors: ["Yunho Jin", "Gu-Yeon Wei", "David Brooks"]
tags: ["LLM", "Test Time Compute", "Energy Efficiency", "Reasoning", "Inference"]
institution: ["Harvard University"]
description: "本文系统评估了测试时计算（TTC）在大型语言模型推理中的精度-能耗权衡，揭示其在复杂推理任务上的潜力，并强调输出序列长度和任务难度对资源优化的重要性。"
---

> **Summary:** 本文系统评估了测试时计算（TTC）在大型语言模型推理中的精度-能耗权衡，揭示其在复杂推理任务上的潜力，并强调输出序列长度和任务难度对资源优化的重要性。 

> **Keywords:** LLM, Test Time Compute, Energy Efficiency, Reasoning, Inference

**Authors:** Yunho Jin, Gu-Yeon Wei, David Brooks

**Institution(s):** Harvard University


## Problem Background

大型语言模型（LLMs）通过参数规模和训练数据扩展提升性能，但面临收益递减和高能耗问题，尤其是在推理阶段（行业报告显示推理占 AI 能耗的 60%-90%）；
本文提出测试时计算（Test-time Compute, TTC），即在推理时投入额外计算资源，以探索是否能在精度-能耗权衡上优于传统规模扩展，同时促进可持续 AI 发展。

## Method

*   **核心策略:** 测试时计算（TTC）通过在推理阶段分配额外计算资源提升模型性能，避免增加预训练成本，具体包括两种方法：
    *   **生成多个候选答案（MV）:** 通过并行采样生成多个输出（如 5 个样本），并通过简单聚合（如多数投票）选择最终答案，增加输入和输出 token 数量，主要影响预填充（prefill）和解码（decode）阶段的计算负载。
    *   **自我精炼推理（RT）:** 通过迭代推理或生成更多推理 token 精炼输出，显著增加输出序列长度，主要影响解码阶段的内存带宽需求（由于 KV 缓存大小线性增长）。
*   **实验设计:** 使用 Qwen2.5 模型系列（1.5B、7B、14B、32B 参数）作为测试对象，在 NVIDIA A100 GPU 上运行，借助 Evalchemy 和 SGLang 工具执行推理；
    通过 NVIDIA Management Library (NVML) 监控功率并计算总能耗，评估数学推理、代码生成和常识任务三大类基准测试的表现。
*   **分析维度:** 研究输出序列长度、任务难度和批量大小对精度和能耗的影响，例如通过逐步调整最大序列长度（步长为 3277 token）观察 Pareto 前沿，以及按难度级别（易、中、难）划分任务以分析资源分配效率。

## Experiment

*   **有效性:** TTC 在数学和代码生成等复杂推理任务上展现出更好的精度-能耗权衡，例如 1.5B 模型使用 RT 策略后在数学基准测试中精度提升 34.3%，远超规模扩展（1.5B 到 7B 仅提升 4.8%）；
    小模型结合 TTC 可媲美甚至超越大模型，如 1.5B RT 模型在 MATH500 上以 135.22Wh 能耗达到 83.2% 精度，优于 32B Base 模型（45.7% 精度，424.54Wh）。
*   **能耗代价:** TTC 可能导致能耗剧增，例如 7B 模型在代码基准上使用 RT 时能耗增加 97.48 倍，原因是输出 token 数量激增（平均 4.4 倍，极端达 46.26 倍）；
    MV 策略平均能耗增加 2.01-4.69 倍，RT 则为 1.08-10.4 倍。
*   **任务差异:** 在常识任务上 TTC 效果有限甚至负面，因其更依赖事实知识而非推理。
*   **实验合理性:** 实验设置全面，覆盖多种模型规模、任务类型和难度级别，运行 10 次以确保数据稳定性，分析了输出长度和批量大小的影响；
    但未考虑硬件生产、训练能耗及优化技术（如量化）的综合影响，存在一定局限性。

## Further Thoughts

输出序列长度作为模型理解度和任务难度的指标，这一发现启发我们可以在推理时根据长度动态调整计算资源，例如设置长度阈值进行早期退出（early exit），以减少不必要能耗；
此外，任务难度与能耗的相关性提示难度感知的模型选择策略（difficulty-aware model selection），即根据任务复杂度动态分配小模型或 TTC 增强模型，可能在云服务和数据中心中显著提升资源效率。