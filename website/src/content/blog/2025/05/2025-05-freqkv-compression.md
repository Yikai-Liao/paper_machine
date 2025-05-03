---
title: "FreqKV: Frequency Domain Key-Value Compression for Efficient Context Window Extension"
pubDatetime: 2025-05-01T14:53:12+00:00
slug: "2025-05-freqkv-compression"
type: "arxiv"
id: "2505.00570"
score: 0.6613495617425725
author: "grok-3-latest"
authors: ["Jushi Kai", "Boyi Zeng", "Yixuan Wang", "Haoli Bai", "Bo Jiang", "Zhouhan Lin"]
tags: ["LLM", "Context Extension", "KV Cache", "Frequency Domain", "Compression"]
institution: ["Shanghai Jiao Tong University (LUMIA Lab)", "Huawei Noah’s Ark Lab", "Shanghai Jiao Tong University"]
description: "FreqKV 通过频率域中 KV 缓存的迭代压缩，在不引入额外参数的情况下高效扩展上下文窗口，显著降低计算和内存成本，同时保持模型性能。"
---

> **Summary:** FreqKV 通过频率域中 KV 缓存的迭代压缩，在不引入额外参数的情况下高效扩展上下文窗口，显著降低计算和内存成本，同时保持模型性能。 

> **Keywords:** LLM, Context Extension, KV Cache, Frequency Domain, Compression

**Authors:** Jushi Kai, Boyi Zeng, Yixuan Wang, Haoli Bai, Bo Jiang, Zhouhan Lin

**Institution(s):** Shanghai Jiao Tong University (LUMIA Lab), Huawei Noah’s Ark Lab, Shanghai Jiao Tong University


## Problem Background

大型语言模型（LLMs）在处理长上下文任务时，受限于预设上下文窗口大小，且自注意力机制的计算成本随序列长度呈二次方增长，KV 缓存内存需求线性增加；现有方法在推理时压缩 KV 缓存或扩展上下文窗口往往导致性能下降或效率不足，亟需一种高效的上下文扩展方法来平衡性能与计算成本。

## Method

* **核心思想**：利用频率域中 KV 状态能量主要集中于低频分量的特性，通过滤除高频分量压缩 KV 缓存，以减少信息损失并提升计算效率，同时实现上下文窗口扩展。
* **具体实现**：
  * 使用离散余弦变换（DCT）将 KV 状态从时域（序列维度）转换为频域，分析其能量分布。
  * 根据预设保留比例（retaining ratio, γ），保留低频分量，滤除高频分量以实现压缩。
  * 使用逆 DCT（IDCT）将压缩后的频域信号转换回时域，并通过幅度校正恢复原始信号强度。
  * 采用迭代压缩策略：当 KV 缓存达到预设上下文窗口大小时触发压缩，保留低频分量后继续添加新 token，早期 token 经历更多次压缩，近期 token 压缩较少。
  * 特殊处理注意力沉点（attention sinks），即初始 token 不被压缩，以保留重要信息。
* **优势**：无需引入额外参数或架构修改，仅通过少量微调即可让模型适应压缩后的 KV 缓存；压缩操作在推理时动态进行，计算开销极小。

## Experiment

* **有效性**：在长上下文语言建模任务中，FreqKV 在 PG-19 数据集上困惑度（PPL）表现优异，与 LongLoRA 等方法相比，在扩展上下文（如 8K-64K）时性能稳定；在 LongBench 基准测试中，FreqKV 在多个长上下文理解任务上达到最优（SOTA），尤其在 LLaMA-2-chat 上表现突出；在 Needle-in-a-Haystack 测试中，显著优于 PyramidKV 和 Dropping 方法。
* **局限性**：在 Proof-pile 数据集的长上下文上，FreqKV 略逊于使用完整 KV 缓存的 LongLoRA，可能在结构化数据（如数学文本）的信息保留上仍有改进空间。
* **效率**：FreqKV 的解码时间随序列长度近似线性增长，相比完整 KV 缓存的二次方增长具有显著优势；压缩操作的额外计算开销极小（在 32K 长度下仅占总时间的 0.64%）。
* **实验设置**：实验覆盖 LLaMA-2-7b 和 LLaMA-3-8b 模型，涉及长上下文语言建模和理解任务，测试了不同上下文长度（4K-64K），对比了多种基线方法，设置全面且合理。

## Further Thoughts

FreqKV 的频率域压缩思路启发了我，或许可以进一步探索不同频率分量的语义重要性，根据任务类型动态调整保留比例；此外，注意力沉点不压缩策略提示是否可以结合注意力分数设计更精细的选择性压缩；频率域压缩与其他技术（如量化或稀疏化）的结合也可能进一步提升效率。