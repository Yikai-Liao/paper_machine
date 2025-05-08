---
title: "JTCSE: Joint Tensor-Modulus Constraints and Cross-Attention for Unsupervised Contrastive Learning of Sentence Embeddings"
pubDatetime: 2025-05-05T05:09:21+00:00
slug: "2025-05-tensor-cross-attention-embedding"
type: "arxiv"
id: "2505.02366"
score: 0.5257884144117333
author: "grok-3-latest"
authors: ["Tianyu Zong", "Hongzhu Yi", "Bingkang Shi", "Yuanxiang Wang", "Jungang Xu"]
tags: ["Sentence Embeddings", "Contrastive Learning", "Tensor Constraints", "Cross-Attention", "BERT Optimization"]
institution: ["University of Chinese Academy of Sciences", "Institute of Information Engineering, Chinese Academy of Sciences"]
description: "本文提出 JTCSE 框架，通过联合张量模长约束和跨注意力机制，显著提升无监督句子嵌入在语义文本相似性任务中的性能，成为当前 SOTA。"
---

> **Summary:** 本文提出 JTCSE 框架，通过联合张量模长约束和跨注意力机制，显著提升无监督句子嵌入在语义文本相似性任务中的性能，成为当前 SOTA。 

> **Keywords:** Sentence Embeddings, Contrastive Learning, Tensor Constraints, Cross-Attention, BERT Optimization

**Authors:** Tianyu Zong, Hongzhu Yi, Bingkang Shi, Yuanxiang Wang, Jungang Xu

**Institution(s):** University of Chinese Academy of Sciences, Institute of Information Engineering, Chinese Academy of Sciences


## Problem Background

无监督句子嵌入的对比学习中，现有方法（如 SimCSE）主要通过 InfoNCE 损失约束正负样本在高维语义空间中的方向分布，但忽略了语义表示张量的模长特征，导致正样本对齐不足；同时，BERT 类模型存在注意力沉没现象，[CLS] 标记无法有效聚合全局语义信息，影响嵌入质量；此外，传统集成学习方法推理开销巨大或依赖非自主训练，存在效率和公平性问题。

## Method

* **核心思想**：提出 JTCSE 框架，通过联合张量模长约束和跨注意力机制，增强无监督对比学习中正样本的对齐能力，同时优化 BERT 类模型对 [CLS] 标记的注意力分配，降低推理开销。
* **张量模长约束**：设计训练目标 *L_TMC*，基于 Pooler 输出（而非受 LayerNorm 影响的最后隐藏状态）计算正样本对的模长差异，促使正样本在高维空间中方向和模长均接近；具体通过两个子目标实现：正样本差向量模长尽量小，各自模长尽量大。
* **跨注意力机制**：在双塔集成模型中引入跨注意力层（CAEL），在特定编码层（如每隔 k 层）让两个子编码器交互信息；一个编码器的注意力权重（Query 和 Key 计算）用于加权另一个编码器的 Value 张量，丰富 [CLS] 标记的语义信息，缓解注意力沉没。
* **损失函数设计**：结合 InfoNCE 损失（继续训练子编码器）、交互约束 InfoNCE（ICNCE，增强双塔间正样本对齐）和交互张量模长约束（ICTM），形成综合损失函数。
* **推理优化**：相比 EDFSE 的六塔结构，JTCSE 仅用两塔降低推理开销，并通过知识蒸馏生成单塔模型（JTCSE D）以提升实用性。

## Experiment

* **有效性**：JTCSE 在 7 个语义文本相似性（STS）任务上平均性能达到 79.70（BERT-base）和 79.94（RoBERTa-base），超越所有开源基线（如 SimCSE 76.25, EDFSE 79.04），成为 SOTA；蒸馏后的 JTCSE D 也表现出色（79.89 和 79.91）。
* **全面性**：在 MTEB 框架的 130+ 零样本下游任务（包括文本分类、检索、重新排序、多语言 STS 等）中，JTCSE 及其衍生模型整体表现最佳，尤其在检索任务（如 MAP@10, NDCG@10）上显著优于基线。
* **推理效率**：JTCSE 推理开销（10.90 GMAC）仅为 EDFSE（32.70 GMAC）的三分之一，但性能更优，单位推理开销性能（η）最高。
* **稳定性**：通过随机种子测试，JTCSE 双塔结构性能波动小，表现出较强稳定性。
* **消融实验**：验证了模长约束和跨注意力各自贡献，3 个 CAEL 层效果最佳；损失函数各部分协同作用显著，缺少任一部分均导致性能下降。

## Further Thoughts

张量模长约束可推广至多模态学习，通过约束不同模态表示的模长特征增强跨模态对齐；跨注意力机制可探索动态调整策略，根据任务需求自适应选择交互层；注意力沉没问题可能影响其他依赖 [CLS] 标记的任务，值得研究其在分类等任务中的表现及解决方案；Pooler 层等未充分利用结构的预训练信息挖掘具有潜力，可在其他预训练模型任务中进一步探索。