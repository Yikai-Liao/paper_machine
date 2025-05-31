---
title: "Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design"
pubDatetime: 2025-05-28T09:55:08+00:00
slug: "2025-05-speculative-quantization-hierarchy"
type: "arxiv"
id: "2505.22179"
score: 0.6966113580734965
author: "grok-3-latest"
authors: ["Yudi Zhang", "Weilin Zhao", "Xu Han", "Tiejun Zhao", "Wang Xu", "Hailong Cao", "Conghui Zhu"]
tags: ["LLM", "Speculative Decoding", "Quantization", "Inference Optimization", "Memory Efficiency"]
institution: ["Harbin Institute of Technology", "Tsinghua University"]
description: "本文提出了一种分层推测解码框架，通过小型中间模型解耦草稿生成与验证过程，显著提升了4位量化大型语言模型的推理效率，实现了2.78倍的加速。"
---

> **Summary:** 本文提出了一种分层推测解码框架，通过小型中间模型解耦草稿生成与验证过程，显著提升了4位量化大型语言模型的推理效率，实现了2.78倍的加速。 

> **Keywords:** LLM, Speculative Decoding, Quantization, Inference Optimization, Memory Efficiency

**Authors:** Yudi Zhang, Weilin Zhao, Xu Han, Tiejun Zhao, Wang Xu, Hailong Cao, Conghui Zhu

**Institution(s):** Harbin Institute of Technology, Tsinghua University


## Problem Background

大型语言模型（LLMs）在单批次推理中面临内存带宽瓶颈，推测解码（Speculative Decoding）和量化（Quantization）是两种常见优化技术，但将两者结合时，特别是在4位权重量化模型上，推测解码的计算开销（如树状草稿验证）会抵消量化的内存优势，导致加速效果有限；本文旨在探究两者的兼容性问题，并解决这一冲突。

## Method

* **核心思想**：针对 W4A16 量化模型，设计一个分层推测解码框架（Hierarchical Speculative Decoding Framework），通过引入小型中间模型，将推测解码的草稿生成与验证过程解耦，兼顾高效草稿生成和低计算开销的验证。
* **具体实现**：
  * **计算密集型草稿阶段**：采用 EAGLE-2 的轻量级草稿模型生成树状草稿（Tree-Style Draft），并由一个小型中间模型（如 W4A16 Llama-3-8B）进行验证，将树状草稿转换为序列草稿（Sequence Draft），利用树状结构的草稿生成效率。
  * **内存高效验证阶段**：目标模型（如 W4A16 Llama-3-70B）对序列草稿进行验证，由于是序列形式，计算开销较低，能够充分发挥4位量化模型的内存访问优势。
* **关键创新**：通过小型中间模型作为桥梁，避免了直接在目标模型上进行高开销的树状验证，同时保留了推测解码的高效草稿生成能力，实现了计算与内存优化的正交性。

## Experiment

* **有效性**：在 W4A16 Llama-3-70B 模型上，提出的分层框架（HierSpec）实现了平均 2.78 倍的加速，相比 EAGLE-2 提升了 1.31 倍，相比 Vanilla SP 提升了 1.21 倍，显著优于基线方法。
* **全面性**：实验覆盖了多种任务（机器翻译、多轮对话、数学推理等）和数据集（SpecBench），在 NVIDIA A100 和 RTX 3090 两种硬件上测试，考虑了不同草稿长度和树大小的影响，设置较为合理。
* **局限性**：实验未涉及长上下文任务和 KV 缓存量化，可能限制了结果在某些场景下的适用性。
* **分析深度**：通过对比不同量化精度（W8A8, W4A16, W4A8）和推测解码配置，揭示了树状验证的计算开销是4位量化模型兼容性差的主要原因，为方法设计提供了数据支持。

## Further Thoughts

分层框架的模块化设计启发了我，是否可以通过联合训练或知识蒸馏增强中间模型与目标模型的对齐性，以进一步提升性能？此外，是否可以根据任务类型或硬件特性动态调整草稿长度和树大小，实现自适应优化？这种解耦计算与内存优化的思想是否能推广到其他领域，如混合精度训练或模型剪枝？