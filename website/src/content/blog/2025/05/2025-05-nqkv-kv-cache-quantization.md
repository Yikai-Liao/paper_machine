---
title: "NQKV: A KV Cache Quantization Scheme Based on Normal Distribution Characteristics"
pubDatetime: 2025-05-22T04:23:19+00:00
slug: "2025-05-nqkv-kv-cache-quantization"
type: "arxiv"
id: "2505.16210"
score: 0.7932237502809838
author: "grok-3-latest"
authors: ["Zhihang Cai", "Xingjun Zhang", "Zhendong Tan", "Zheng Wei"]
tags: ["LLM", "KV Cache", "Quantization", "Memory Optimization", "Inference"]
institution: ["Xi'an Jiaotong University"]
description: "本文提出 NQKV 算法，利用 KV Cache 的正态分布特性进行 4 位分块量化，显著减少内存占用并提升吞吐量，同时保持模型精度。"
---

> **Summary:** 本文提出 NQKV 算法，利用 KV Cache 的正态分布特性进行 4 位分块量化，显著减少内存占用并提升吞吐量，同时保持模型精度。 

> **Keywords:** LLM, KV Cache, Quantization, Memory Optimization, Inference

**Authors:** Zhihang Cai, Xingjun Zhang, Zhendong Tan, Zheng Wei

**Institution(s):** Xi'an Jiaotong University


## Problem Background

大型语言模型（LLM）在推理过程中，注意力机制的键值对缓存（KV Cache）随着批次大小和上下文长度的增加，内存占用急剧上升，成为部署的主要瓶颈；现有量化方法在将 KV Cache 量化为 4 位以下时会导致显著精度下降，亟需一种低位量化方案以减少内存占用并维持模型性能。

## Method

* **核心思想**：基于 KV Cache 中元素在 token 维度及块内符合正态分布的特性，提出 NQKV 算法，通过分块量化与正态分布数据类型实现低位量化，减少内存占用。
* **具体实现**：
  - **数据分布分析**：通过 Q-Q 图和 D’Agostino-Pearson 检验，确认 KV Cache 的键和值在 token 维度及固定块大小内符合正态分布。
  - **分块量化**：将每个 token 的数据按固定块大小分割（例如 256），在每个块内单独进行 4 位 Normal Float（NF4）量化，通过分位点选择最小化量化误差，限制误差传播。
  - **流式处理兼容**：在预填充阶段对初始键值对量化并存储索引到 KV Cache；在解码阶段对新生成的键值对进行量化并追加到缓存，计算前反量化恢复浮点值，适配 KV Cache 的追加式特性。
  - **填充优化**：在 token 维度上进行填充以适配 GPU 矩阵乘法优化，选择在反量化后填充以减少存储和计算开销。
* **特点**：无需模型重新训练或微调，与现有权重和激活量化方法正交，可联合使用。

## Experiment

* **精度表现**：NQKV 将 KV Cache 量化为 4 位后，精度损失极小，例如 OPT-1.3B 平均精度下降仅 0.7%，OPT-6.7B 和 OPT-13B 损失更小，甚至在部分任务上略有提升，表明方法对模型输出质量影响微乎其微。
* **内存与吞吐量提升**：NQKV 使 OPT-6.7B 支持 4 倍批次大小或 2.5 倍序列长度（相比 FP16），相比 SmoothQuant 提升 2 倍批次大小或 1.5 倍序列长度；在 OPT-30B 上，NQKV 实现 9.3 倍吞吐量提升，额外节省 60%-80% 内存。
* **实验设置合理性**：实验覆盖 OPT 模型家族（125M-30B），测试多种零-shot 任务和不同批次大小/序列长度场景，与 FP16 和 SmoothQuant 对比充分，验证了方法的独立效果和兼容性；吞吐量略低于 SmoothQuant（因反量化开销），但相比不启用 KV Cache 仍有显著加速。

## Further Thoughts

NQKV 基于数据分布特性设计量化的思路启发我们，可以针对其他深度学习模块的统计特性定制量化方案；存储与计算分离的策略（使用存储型数据类型如 NF4）可在高内存场景中借鉴；流式处理兼容的设计对实时推理任务有参考价值；未来可探索动态调整块大小或量化位数的自适应量化，或结合硬件设计专用正态分布数据类型以减少反量化开销。