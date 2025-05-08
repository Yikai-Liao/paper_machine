---
title: "Large Language Model Partitioning for Low-Latency Inference at the Edge"
pubDatetime: 2025-05-05T10:16:16+00:00
slug: "2025-05-llm-edge-partitioning"
type: "arxiv"
id: "2505.02533"
score: 0.6305985267755974
author: "grok-3-latest"
authors: ["Dimitrios Kafetzis", "Ramin Khalili", "Iordanis Koutsopoulos"]
tags: ["LLM", "Edge Computing", "Transformer Partitioning", "Resource Allocation", "Low-Latency Inference"]
institution: ["Athens University of Economics and Business", "Huawei European Research Center"]
description: "本文提出一种资源感知的 Transformer 分区算法，通过注意力头级别的细粒度分区和动态块迁移，显著降低边缘环境下大型语言模型的推理延迟并优化内存使用。"
---

> **Summary:** 本文提出一种资源感知的 Transformer 分区算法，通过注意力头级别的细粒度分区和动态块迁移，显著降低边缘环境下大型语言模型的推理延迟并优化内存使用。 

> **Keywords:** LLM, Edge Computing, Transformer Partitioning, Resource Allocation, Low-Latency Inference

**Authors:** Dimitrios Kafetzis, Ramin Khalili, Iordanis Koutsopoulos

**Institution(s):** Athens University of Economics and Business, Huawei European Research Center


## Problem Background

大型语言模型（LLMs）在边缘设备上的推理面临低延迟挑战。由于边缘设备资源有限，而 LLMs 的自回归解码特性导致推理过程中内存和计算需求（尤其是 K/V 缓存）随 token 数量增加而持续增长，传统静态层级分区方法容易引发内存超载和高延迟。本文旨在通过更细粒度的分区和动态资源分配，降低推理延迟，充分利用边缘设备的集体资源。

## Method

* **核心思想**：提出一种资源感知的 Transformer 架构分区算法，通过注意力头级别的细粒度分区和动态块迁移，优化边缘环境下的推理延迟。
* **细粒度分区**：将多头注意力机制（MHA）中的每个注意力头及其对应的 K/V 缓存作为一个独立单元，与前馈网络（FFN）和输出投影（proj）块一起分配到不同边缘设备上。这种方法允许注意力头并行执行，提升资源利用率。
* **动态迁移**：算法在推理过程中以固定间隔（例如每生成一个 token）更新分区决策，由集中式控制器根据设备当前的内存、计算能力和网络带宽，动态决定块的分配和迁移，采用‘短视’（myopic）策略，仅依赖即时资源信息。
* **算法实现细节**：控制器通过评分函数（综合考虑内存占用、计算负载和通信开销）为每个块选择最合适的设备；当资源超载时，通过回溯机制和块迁移解决冲突；设置时间和迭代上限以确保算法在有限时间内完成。
* **关键优势**：相比传统层级分区，注意力头级别的分区更灵活，能适应 K/V 缓存的动态增长，同时动态迁移避免了设备过载。

## Experiment

* **小规模场景（3-5 个设备）**：在生成少量 token 的情况下，与精确求解器（optimal solver）相比，提出的资源感知算法延迟仅高出 15-20%，显著优于其他基线方法（如 Greedy、Round-Robin）的 40-60% 差距，表明算法接近最优。
* **中规模场景（25 个设备）**：在生成高达 1000 个 token 的情况下，推理延迟显著低于现有方法（如 EdgeShard 和 Galaxy），最高实现 9-10 倍加速；内存使用也更高效，尤其在 K/V 缓存增长时，控制在较低水平（约 6GB vs. 其他方法的 7GB 以上）。
* **实验设置合理性**：实验通过自定义模拟器建模设备异构性（内存和计算能力从对数正态分布采样）和网络带宽波动，设置每 token 更新一次分区决策的‘最坏情况’场景，充分测试了算法在高迁移频率下的表现。
* **局限性**：实验基于单层解码器，推广到多层 Transformer 模型的效果需进一步验证；随着设备数量增加，协调和调度开销上升，可能影响大规模场景的表现。

## Further Thoughts

注意力头级别的细粒度分区和动态资源分配是一个极具启发性的想法，不仅适用于 LLMs，也可能推广到其他深度学习模型（如视觉 Transformer 或多模态模型）在边缘环境下的分布式推理；此外，‘短视’策略可以通过引入有限预测（如基于历史资源使用模式）进一步优化；另一个方向是结合能量约束或请求负载预测，探索更真实的边缘场景应用。