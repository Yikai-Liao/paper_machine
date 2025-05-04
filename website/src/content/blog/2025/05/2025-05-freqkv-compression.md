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
institution: ["Shanghai Jiao Tong University", "Huawei Noah’s Ark Lab"]
description: "本文提出 FreqKV 方法，通过频率域中 KV 状态的低频分量保留实现高效压缩，显著扩展大型语言模型的上下文窗口，同时保持性能和降低计算与内存开销。"
---

> **Summary:** 本文提出 FreqKV 方法，通过频率域中 KV 状态的低频分量保留实现高效压缩，显著扩展大型语言模型的上下文窗口，同时保持性能和降低计算与内存开销。 

> **Keywords:** LLM, Context Extension, KV Cache, Frequency Domain, Compression

**Authors:** Jushi Kai, Boyi Zeng, Yixuan Wang, Haoli Bai, Bo Jiang, Zhouhan Lin

**Institution(s):** Shanghai Jiao Tong University, Huawei Noah’s Ark Lab


## Problem Background

大型语言模型（LLMs）在处理长上下文时面临内存和计算效率的挑战：KV 缓存的内存需求随上下文长度线性增长，自注意力机制的计算复杂度随序列长度呈二次方增长，导致上下文窗口扩展在微调和推理阶段变得困难；现有方法在扩展到更长上下文时往往造成性能下降，因此需要一种高效的 KV 缓存压缩方法，既能降低资源开销，又能尽量维持模型性能。

## Method

* **核心思想**：利用频率域中 KV 状态能量主要集中在低频分量的特性，通过滤除高频分量对 KV 缓存进行压缩，从而在不引入额外参数或架构修改的情况下实现高效的上下文窗口扩展。
* **具体实现**：
  * 使用离散余弦变换（DCT）将 KV 状态从时域（序列维度）转换为频率域，分析其能量分布。
  * 根据预设的保留比例（γ），保留低频分量，滤除高频分量，压缩 KV 缓存至固定大小。
  * 通过逆 DCT（IDCT）将压缩后的频率域信号转换回时域，并进行幅度校正以恢复原始信号强度。
  * 采用迭代压缩机制：当 KV 缓存达到预设上下文窗口大小时触发压缩，压缩后的缓存与后续 token 继续累积，直到再次达到上限；早期 token 被压缩次数较多，近期 token 压缩较少。
  * 保留初始的注意力沉点（attention sink）token 不被压缩，以避免关键信息丢失。
* **关键特点**：FreqKV 不需要额外的压缩模块，仅通过微调即可让模型适应压缩后的 KV 缓存，同时支持微调和推理阶段的上下文扩展。

## Experiment

* **有效性**：在长上下文语言建模任务中，FreqKV 在短上下文（2K、4K）几乎无性能下降，在长上下文（8K 及以上）与使用完整 KV 缓存的 LongLoRA 相比表现相当，甚至在 PG-19 数据集上超越 LongLoRA 和全微调（Full FT）；在长上下文理解任务（LongBench 和 Needle-in-a-Haystack）中，FreqKV 在多个子任务上取得 SOTA 表现，平均准确率显著高于其他 KV 压缩方法。
* **效率提升**：FreqKV 的解码时间随序列长度近似线性增长，而完整 KV 缓存的解码时间呈二次方增长；压缩操作的额外计算开销极小（在 32K 长度下仅占总时间的 0.64%）。
* **实验设置合理性**：实验基于 LLaMA-2-7B 和 LLaMA-3-8B 模型，涵盖语言建模和理解任务，数据集（如 RedPajama、PG-19、LongBench）选择合理，评估指标（如困惑度 PPL、准确率）全面，能够充分验证方法的性能和效率。

## Further Thoughts

FreqKV 利用频率域能量分布进行 KV 缓存压缩的思路非常具有启发性，未来可以探索将其推广到多模态模型中的特征压缩；此外，是否可以通过动态评估 token 重要性（如基于注意力分数或语义贡献）设计自适应压缩策略？或者结合量化、稀疏化等技术进一步提升压缩效率？另外，针对不同任务特性（如历史信息或近期信息的重要性），是否可以调整压缩分布以优化性能？