---
title: "Trainable Dynamic Mask Sparse Attention"
pubDatetime: 2025-08-04T07:05:15+00:00
slug: "2025-08-dynamic-mask-attention"
type: "arxiv"
id: "2508.02124"
score: 0.6631676673038202
author: "grok-3-latest"
authors: ["Jingze Shi", "Yifan Wu", "Bingheng Wu", "Yiran Peng", "Liangdong Wang", "Guang Liu", "Yuyu Luo"]
tags: ["LLM", "Sparse Attention", "Long Context", "Efficiency", "Trainable Model"]
institution: ["HKUST(GZ)", "BAAI", "SmallDoges"]
description: "本文提出动态掩码注意力（DMA），一种可训练的稀疏注意力机制，通过内容感知和位置感知稀疏计算，显著提升大型语言模型在长上下文任务中的效率和性能。"
---

> **Summary:** 本文提出动态掩码注意力（DMA），一种可训练的稀疏注意力机制，通过内容感知和位置感知稀疏计算，显著提升大型语言模型在长上下文任务中的效率和性能。 

> **Keywords:** LLM, Sparse Attention, Long Context, Efficiency, Trainable Model

**Authors:** Jingze Shi, Yifan Wu, Bingheng Wu, Yiran Peng, Liangdong Wang, Guang Liu, Yuyu Luo

**Institution(s):** HKUST(GZ), BAAI, SmallDoges


## Problem Background

大型语言模型（LLMs）在处理长上下文任务（如长文档推理、代码生成、多轮对话）时，标准自注意力机制因其二次方计算复杂度（O(n²)）导致计算和内存开销巨大，限制了模型扩展到更长序列的能力。
现有稀疏注意力机制虽然提升了效率，但常因静态模式或信息丢失而无法在效率和建模能力之间取得平衡，论文旨在解决如何设计一种高效且有效的注意力机制，显著降低计算复杂度同时保留长上下文中的关键信息。

## Method

*   **核心思想：** 提出动态掩码注意力（Dynamic Mask Attention, DMA），一种可训练的稀疏注意力机制，通过内容感知和位置感知的双重稀疏性，在保留完整信息的同时大幅降低计算复杂度。
*   **内容感知动态稀疏掩码：** 从值（Value）表示中动态生成掩码，决定每个注意力头应关注的历史token；通过可学习参数（如采样权重矩阵Δ和门控参数A）计算动态权重，利用零阶保持方法确保权重稳定性，并结合top-k操作选择最重要的token，生成自适应掩码。
*   **位置感知稀疏注意力计算：** 基于动态掩码，跳过被掩码区域（注意力权重为0的位置）的计算，减少无效操作；保留完整的KV缓存以确保信息完整性，避免固定状态压缩带来的信息瓶颈。
*   **硬件优化与可微分设计：** 设计专用计算内核，支持在GPU上高效跳过无效计算，结合Flash Attention的块计算策略提升实际速度；通过完全可微分设计，确保训练和推理阶段的稀疏策略一致，支持端到端学习。
*   **关键优势：** DMA将稀疏性嵌入模型架构，避免后处理稀疏化带来的性能下降，同时在多头注意力中为每个头生成独特的掩码结构，最大化子空间利用率。

## Experiment

*   **有效性：** 在SmolLMCorpus数据集上，DMA在不同参数规模（80M到1.7B）的困惑度测试中优于多头注意力（MHA）、滑动窗口注意力（SWA）、多头潜在注意力（MLA）和原生稀疏注意力（NSA）。
*   **长上下文任务表现：** 在多查询关联回忆任务中，DMA在长序列信息检索上表现优异，表明其能有效识别关键token；在‘针在干草堆’任务中，DMA展现出更强的长度外推能力，超出预训练序列长度时性能下降较小。
*   **效率提升：** 推理速度上，DMA对长序列（4096及以上）的加速显著，CUDA实现比MHA快10倍以上，尽管短序列上因额外采样略慢于SWA。
*   **下游任务：** 在1.7B参数模型上，DMA在多个基准测试（如MMLU、TriviaQA）中优于MHA和NSA，零-shot和five-shot设置下均表现最佳。
*   **实验设置合理性：** 实验覆盖了模型规模、任务类型（困惑度、检索、基准测试）、硬件实现等多维度，设置全面，结果可信，充分验证了DMA在效率和性能上的双重优势。

## Further Thoughts

DMA的可训练稀疏性设计启发我们可以在其他深度学习模块中探索‘原生稀疏性’，将其嵌入架构以避免后处理性能损失；其内容与位置双重稀疏策略提示针对不同任务需求设计多层次稀疏模式；此外，DMA在长度外推上的优势表明稀疏注意力可能为位置编码提供新思路，值得探索其与位置信息的深度结合；最后，论文提到的多模态扩展潜力启发我们思考如何为跨模态长距离依赖设计特定稀疏模式。