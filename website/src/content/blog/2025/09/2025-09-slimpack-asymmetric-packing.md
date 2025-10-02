---
title: "SlimPack: Fine-Grained Asymmetric Packing for Balanced and Efficient Variable-Length LLM Training"
pubDatetime: 2025-09-30T13:37:48+00:00
slug: "2025-09-slimpack-asymmetric-packing"
type: "arxiv"
id: "2509.26246"
score: 0.6105971861089065
author: "grok-3-latest"
authors: ["Yuliang Liu", "Wei Zhang", "Guohao Wu", "Qianchao Zhu", "Chenyu Wang", "Shenglong Zhang", "Zhouyang Li"]
tags: ["LLM", "Distributed Training", "Load Balancing", "Memory Efficiency", "Pipeline Parallelism"]
institution: ["Kuaishou Technology, Kling Infra"]
description: "SlimPack 通过细粒度切片和不对称分区，显著提升了变长序列大型语言模型训练的效率和负载均衡能力，实验证明其在长上下文场景下实现了高达 2.8 倍的吞吐量提升。"
---

> **Summary:** SlimPack 通过细粒度切片和不对称分区，显著提升了变长序列大型语言模型训练的效率和负载均衡能力，实验证明其在长上下文场景下实现了高达 2.8 倍的吞吐量提升。 

> **Keywords:** LLM, Distributed Training, Load Balancing, Memory Efficiency, Pipeline Parallelism

**Authors:** Yuliang Liu, Wei Zhang, Guohao Wu, Qianchao Zhu, Chenyu Wang, Shenglong Zhang, Zhouyang Li

**Institution(s):** Kuaishou Technology, Kling Infra


## Problem Background

大型语言模型（LLM）训练面临输入序列长度极端异构性导致的效率低下问题，这种长尾分布特性使得少量超长序列占据大部分计算负载，传统打包策略和静态并行方法无法有效应对，造成硬件利用率低下、级联式负载不平衡以及内存和通信瓶颈；此外，前向和反向计算成本的不对称性进一步加剧了问题，即使前向阶段平衡，反向阶段仍会出现新的不平衡。

## Method

* **核心思想**：通过将样本分解为细粒度切片（slice-level decomposition），构建均衡的调度单位 MicroPack，并在不引入额外通信开销的前提下，针对前向和反向计算成本的不对称性进行动态优化。
* **细粒度切片与 MicroPack 构建**：长序列被切分为多个小片段，短序列则打包成组，形成计算负载均衡的 MicroPack，避免传统样本级打包中因长序列导致的‘拖延者’（straggler）问题。
* **不对称分区（Asymmetric Partitioning）**：针对前向和反向阶段的计算成本差异，分别设计不同的 MicroPack 配置，确保两个阶段均实现负载均衡，避免因反向阶段重计算（如 FlashAttention）导致的不平衡。
* **两阶段求解器（Two-Phase Solver）**：第一阶段在数据并行（DP）级别分配样本，确保每个 DP 组的计算负载均衡；第二阶段在每个 DP 组内进行切片和打包，通过混合整数线性规划（MILP）优化 MicroPack 配置。
* **DP-Merge 技术**：对于极端超长序列，通过临时合并多个 DP 组并应用上下文并行（Context Parallelism）来分摊计算和内存压力，避免跨节点通信开销。
* **DAG 模拟器**：基于有向无环图（DAG）的高保真模拟器，精确建模任务依赖、内存占用和管道气泡（pipeline bubbles），评估不同调度策略的性能，选择最优配置。

## Experiment

* **性能提升**：在 Llama 模型（7B 到 150B 参数规模）和三个数据集（Common Crawl、GitHub、Wikipedia）上，SlimPack 相比基线 Megatron-LM 实现了高达 2.8 倍的训练吞吐量提升，尤其在长上下文（256K 令牌）场景下表现突出。
* **负载均衡**：通过 violin 图对比，SlimPack 的切片级打包显著降低了前向和反向阶段的计算时间方差，消除了传统打包策略中的长尾延迟问题，特别是在长尾分布数据集（如 Common Crawl）上效果显著。
* **内存效率**：DAG 模拟器的内存预测与实际峰值内存使用高度一致（MAPE 仅 1.6%），确保了调度策略不会导致内存溢出（OOM），同时通过切片级分解降低了峰值内存占用。
* **实验设置合理性**：实验覆盖了多种模型规模、上下文长度（64K 到 256K）、GPU 数量（128 到 256）和批次大小（512 到 2048），充分验证了 SlimPack 的普适性和可扩展性。

## Further Thoughts

SlimPack 的细粒度切片和不对称优化策略启发了我思考如何将这种分解和动态调整的思路应用到其他深度学习任务中，例如图像处理或多模态模型训练，以处理异构数据；此外，DAG 模拟器的高保真预测能力表明基于模拟的调度优化可能成为未来分布式系统设计的重要工具，可以探索与其他技术（如动态并行性调整）结合的可能性。