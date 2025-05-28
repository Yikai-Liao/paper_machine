---
title: "Win Fast or Lose Slow: Balancing Speed and Accuracy in Latency-Sensitive Decisions of LLMs"
pubDatetime: 2025-05-26T04:03:48+00:00
slug: "2025-05-latency-quality-tradeoff"
type: "arxiv"
id: "2505.19481"
score: 0.5295683020120718
author: "grok-3-latest"
authors: ["Hao Kang", "Qingru Zhang", "Han Cai", "Weiyuan Xu", "Tushar Krishna", "Yilun Du", "Tsachy Weissman"]
tags: ["LLM", "Latency Trade-off", "Quantization", "Real-Time Decision", "Adaptive Inference"]
institution: ["Georgia Tech", "UC Berkeley", "Harvard University", "Stanford University"]
description: "本文提出 FPX 自适应混合精度推理框架，通过动态调整模型精度优化延迟-质量权衡，并在高频交易和竞技游戏等延迟敏感任务中显著提升 LLM 性能。"
---

> **Summary:** 本文提出 FPX 自适应混合精度推理框架，通过动态调整模型精度优化延迟-质量权衡，并在高频交易和竞技游戏等延迟敏感任务中显著提升 LLM 性能。 

> **Keywords:** LLM, Latency Trade-off, Quantization, Real-Time Decision, Adaptive Inference

**Authors:** Hao Kang, Qingru Zhang, Han Cai, Weiyuan Xu, Tushar Krishna, Yilun Du, Tsachy Weissman

**Institution(s):** Georgia Tech, UC Berkeley, Harvard University, Stanford University


## Problem Background

大型语言模型（LLMs）在文本生成和复杂推理等任务中表现出色，但其在高频交易和实时竞技游戏等延迟敏感的实时决策任务中的表现受到推理延迟的显著影响；论文旨在系统研究延迟-质量权衡问题，解决如何在动态环境中通过降低延迟提升整体任务性能，而不显著牺牲输出质量。

## Method

* **核心思想**：提出 FPX，一个自适应混合精度推理框架，通过在模型线性层上动态选择 FP8 和 FP4 精度，实现延迟与质量之间的细粒度权衡。
* **具体实现**：
  * **离线校准**：使用校准数据集（如 Wikitext-2）计算每个线性层的量化误差（Relative Error），评估其对低精度（如 FP4）的容忍度。
  * **精度分配**：根据用户指定的压缩比例（γ），将容忍度高的层分配为 FP4 精度，敏感层保留 FP8 精度，确保延迟降低的同时尽量减少质量损失。
  * **优化目标**：专注于 Transformer 架构中占主导地位的矩阵乘法操作（如 QKV 投影、输出投影和前馈层），其他组件（如归一化和注意力机制）保持不变以维持功能正确性。
* **技术优势**：FPX 提供连续的延迟-质量控制，而非传统方法的离散选项（如固定模型大小或静态量化），能够根据任务需求动态调整模型配置。
* **适用性**：方法兼容多种 Transformer 架构，易于部署，且通过离线校准减少在线计算开销。

## Experiment

* **有效性**：在 HFTBench（高频交易基准）上，FPX 结合 14B 参数模型（压缩比例 γ=0.2）实现了最高的日收益率（26.52%），相比纯 FP8（23.14%）和 FP16（17.20%）有显著提升；在 StreetFighter（竞技游戏基准）上，FPX 结合 3B 参数模型（γ=0.3）取得了最高胜率（80%）和 ELO 分数（5.99），优于其他配置。
* **权衡表现**：实验揭示不同任务对延迟和质量的敏感度差异，HFTBench 更依赖高质量决策（大模型表现更优），StreetFighter 更偏向低延迟（小模型更合适），FPX 能通过调整 γ 值找到任务特定的最优权衡点。
* **实验设置**：两个基准设计合理，HFTBench 使用真实历史交易数据模拟市场动态，StreetFighter 基于 DIAMBRA 平台提供实时游戏环境；实验覆盖 1.5B 到 14B 参数的 Qwen2.5 模型系列，测试了 FP16、FP8、FP4 及 FPX 的多种配置，并通过消融研究验证不同压缩比例的效果，设置全面且具说服力。
* **局限性**：当前方法以层级为单位调整精度，未来可能探索更细粒度的控制（如 Token 级），但需更复杂实现。

## Further Thoughts

论文启发我们思考任务特定的延迟-质量权衡需求，提示 LLM 部署需根据应用场景定制优化策略；FPX 的自适应混合精度方法展示了动态调整模型配置的潜力，未来可扩展至其他优化技术（如剪枝或蒸馏）；此外，提出的延迟敏感基准（HFTBench 和 StreetFighter）填补了 LLM 实时任务评估的空白，启发设计更多贴近真实世界的测试环境。