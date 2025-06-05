---
title: "HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference"
pubDatetime: 2025-06-03T07:53:32+00:00
slug: "2025-06-hash-aware-topk-attention"
type: "arxiv"
id: "2506.02572"
score: 0.7367652039675663
author: "grok-3-latest"
authors: ["Ping Gong", "Jiawei Yi", "Shengnan Wang", "Juncheng Zhang", "Zewen Jin", "Ouxiang Zhou", "Ruibo Liu", "Guanbin Xu", "Youhui Bai", "Bowen Ye", "Kun Yuan", "Tong Yang", "Gong Zhang", "Renhai Chen", "Feng Wu", "Cheng Li"]
tags: ["LLM", "Top-k Attention", "Hashing", "Inference Acceleration", "Sparsity"]
institution: ["University of Science and Technology of China", "Institute of Artificial Intelligence, Hefei Comprehensive National Science Center", "Huawei Technologies", "Peking University"]
description: "HATA 提出了一种硬件高效的 Top-k 注意力机制，通过学习哈希技术将查询和键映射为二进制码，以低成本汉明距离计算实现 LLM 推理加速达 7.2 倍，同时保持模型精度。"
---

> **Summary:** HATA 提出了一种硬件高效的 Top-k 注意力机制，通过学习哈希技术将查询和键映射为二进制码，以低成本汉明距离计算实现 LLM 推理加速达 7.2 倍，同时保持模型精度。 

> **Keywords:** LLM, Top-k Attention, Hashing, Inference Acceleration, Sparsity

**Authors:** Ping Gong, Jiawei Yi, Shengnan Wang, Juncheng Zhang, Zewen Jin, Ouxiang Zhou, Ruibo Liu, Guanbin Xu, Youhui Bai, Bowen Ye, Kun Yuan, Tong Yang, Gong Zhang, Renhai Chen, Feng Wu, Cheng Li

**Institution(s):** University of Science and Technology of China, Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, Huawei Technologies, Peking University


## Problem Background

大型语言模型（LLM）的推理过程中，注意力模块是主要的性能瓶颈，尤其是在长序列和大批量场景下，即使采用 KVCache 技术减少冗余计算，内存带宽和计算开销仍显著限制效率。
传统 Top-k 注意力机制通过利用注意力分布的稀疏性减少 KVCache 加载开销，但现有方法在效率和精度之间难以平衡，HATA 旨在解决这一关键问题。

## Method

*   **核心思想:** HATA（Hash-Aware Top-k Attention）通过将学习哈希技术（Learning-to-Hash）集成到 Top-k 注意力机制中，避免对查询-键（Query-Key, qk）分数进行高成本的精确数值估计，而是将查询和键映射为二进制哈希码，通过低成本的汉明距离计算获取相对分数排序，用于高效的 Top-k 选择。
*   **哈希建模与训练:** 设计哈希函数将查询和键向量映射为二进制码，训练目标是最小化相似性损失，确保相似向量在哈希空间中具有较小的汉明距离；同时引入位平衡（Bits Balance）和不相关性（Uncorrelation）约束提升哈希质量；训练数据基于真实数据集中的查询-键对构建，正样本为 Top-10% 高分对，负样本为剩余部分。
*   **HATA 工作流程:** 在预填充（Prefill）阶段，计算并缓存键的哈希码；在解码（Decode）阶段，通过硬件高效的汉明距离计算筛选出 Top-k 键值对，并结合稀疏注意力计算输出；支持多头注意力（Multi-Head Attention）中为每个头训练独立的哈希权重。
*   **硬件优化:** 包括哈希编码的内核融合（Kernel Fusion）、汉明分数计算的高性能算子设计（利用 bitwise_xor 和 bitcount 指令）、以及与 FlashAttention 的集成（融合 Gather 操作），以减少 CPU-GPU 同步和内存访问开销。
*   **关键优势:** 将注意力问题转化为轻量级的序数比较任务（Ordinal Comparison），而非传统的数值回归任务，大幅降低计算和内存成本，同时保持模型精度。

## Experiment

*   **有效性:** HATA 在多个主流 LLM 模型（如 Llama-2-7B, Llama-3.1-8B）和任务（如 LongBench-e, RULER）上实现了高达 7.2 倍的推理加速，相比全注意力机制（Vanilla Full Attention）保持了接近无损的精度（例如在 LongBench-e 上平均精度差距小于 1%）。
*   **优越性:** 相比现有 Top-k 注意力方法（如 Loki, Quest, MagicPIG），HATA 在精度和效率上均有显著优势，尤其在长上下文（如 128K 序列）和大批量场景下，加速比随序列长度增加而提升（例如在 256K 序列下达 6.51 倍加速）。
*   **实验设置合理性:** 实验覆盖了多种模型、数据集和任务类型，测试了不同序列长度（1K 至 256K）和批量大小（1 至 8）的性能，基准方法配置遵循原论文建议，确保对比公平；此外，HATA 的可扩展性在更大模型（如 Qwen2.5-32B）和超长上下文（256K）上得到验证。
*   **开销分析:** HATA 的额外开销主要来自哈希编码和训练，但由于哈希位数较小（默认 128 位），预填充阶段开销不到总计算的 1%，解码阶段的汉明距离计算通过硬件优化大幅降低成本，整体效率提升显著。

## Further Thoughts

HATA 的核心启发在于将注意力机制中的 qk 分数计算简化为序数比较任务，而非高成本的数值回归，这一思路可推广至其他需要稀疏化或近似计算的深度学习模块，如推荐系统或图像检索；此外，学习哈希技术通过轻量级代理模型替代复杂计算的策略，提示了在模型压缩或知识蒸馏中应用类似方法的潜力；最后，HATA 强调硬件协同优化（如内核融合和高效算子设计）的重要性，启发我们在设计 AI 算法时需更多考虑硬件特性和实际部署环境。