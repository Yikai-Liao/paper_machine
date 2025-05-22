---
title: "TinyAlign: Boosting Lightweight Vision-Language Models by Mitigating Modal Alignment Bottlenecks"
pubDatetime: 2025-05-19T09:11:54+00:00
slug: "2025-05-tinyalign-alignment"
type: "arxiv"
id: "2505.12884"
score: 0.6836997316211957
author: "grok-3-latest"
authors: ["Yuanze Hu", "Zhaoxin Fan", "Xinyu Wang", "Gen Li", "Ye Qiu", "Zhichao Yang", "Wenjun Wu", "Kejian Wu", "Yifan Sun", "Xiaotie Deng", "Jin Dong"]
tags: ["Lightweight VLM", "Modal Alignment", "Retrieval Augmentation", "Mutual Information", "Data Efficiency"]
institution: ["Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing, Beihang University", "Hangzhou International Innovation Institute, Beihang University", "Xreal", "Renmin University", "Peking University", "Beijing Academy of Blockchain and Edge Computing (BABEC)"]
description: "TinyAlign 框架通过检索增强生成技术，从内存库中获取相关上下文以提升轻量化视觉-语言模型的模态对齐能力，显著改善性能和数据效率。"
---

> **Summary:** TinyAlign 框架通过检索增强生成技术，从内存库中获取相关上下文以提升轻量化视觉-语言模型的模态对齐能力，显著改善性能和数据效率。 

> **Keywords:** Lightweight VLM, Modal Alignment, Retrieval Augmentation, Mutual Information, Data Efficiency

**Authors:** Yuanze Hu, Zhaoxin Fan, Xinyu Wang, Gen Li, Ye Qiu, Zhichao Yang, Wenjun Wu, Kejian Wu, Yifan Sun, Xiaotie Deng, Jin Dong

**Institution(s):** Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing, Beihang University, Hangzhou International Innovation Institute, Beihang University, Xreal, Renmin University, Peking University, Beijing Academy of Blockchain and Edge Computing (BABEC)


## Problem Background

轻量化视觉-语言模型（Lightweight Vision-Language Models, VLMs）在资源受限场景（如边缘设备）中至关重要，但当前主流的模态对齐方法（冻结预训练的视觉编码器和语言模型，仅训练小型连接模块）在轻量化模型上效果不佳。
原因是小型语言模型（LLM）的表示能力有限，导致模态对齐质量受限，论文从信息论视角分析了这一‘对齐瓶颈’，提出有效互信息（Effective Mutual Information, EMI）受限于模型容量，影响了多模态输入与输出之间的信息流动。

## Method

*   **核心思想**：提出 TinyAlign 框架，灵感来源于检索增强生成（Retrieval-Augmented Generation, RAG），通过从内存库中检索相关上下文增强多模态输入的信息含量，缓解轻量化语言模型的对齐瓶颈。
*   **内存库设计**：从预训练数据集中提取 100K 个图像-文本对，生成紧凑的键-值对（Key-Value Pairs），键用于快速检索（基于注意力机制和均值池化生成低维嵌入），值是压缩后的多模态嵌入（使用 Perceiver 模型预压缩以降低存储和计算开销）。
*   **检索与增强**：在训练和推理时，根据输入图像和指令生成查询键，从内存库中检索 Top-K 个相关嵌入（实验中 Top-5 效果最佳），通过一个可训练的 RAG Connector 将其转化为辅助表示（H_R）。
*   **输入整合**：将原始视觉特征（H_V，通过视觉编码器和主连接模块生成）、检索增强的上下文（H_R）和指令嵌入（H_I）组合为最终输入，送入冻结的轻量化语言模型。
*   **两阶段训练**：包括预训练阶段（仅训练连接模块，冻结视觉编码器和语言模型）和指令微调阶段（联合微调语言模型和连接模块，视觉编码器仍冻结），以适应下游任务。
*   **关键优势**：不依赖外部知识库，直接从训练数据构建内存库，通过压缩和高效检索保持轻量化特性，同时显著提升输入信息含量。

## Experiment

*   **预训练效果**：TinyAlign 显著加速了收敛速度并降低了训练损失，例如在 Phi-2 (2.7B) 上损失降低 16.8%，在 TinyLLaMA (1.1B) 上降低 28.2%，UMAP 可视化显示其嵌入空间对齐更紧凑和语义一致。
*   **指令微调性能**：在多个基准测试（如 VQAv2、GQA、TextVQA）上性能提升明显，例如 TinyLLaMA-1.1B 在 VQAv2 上提升 3.51%，但在极复杂任务（如 MM-Vet）上部分小模型性能略降，表明对复杂任务适应性需优化。
*   **数据效率**：展现极高数据效率，仅用 40% 指令微调数据即可达到基线模型 100% 数据时的性能，尤其在视觉推理和场景文本理解任务上表现突出。
*   **计算开销**：FLOPs 分析显示额外计算负担较小，对推理效率影响可忽略。
*   **实验设置评价**：实验覆盖不同规模模型和多种任务，设置较为全面，但未报告误差条（Error Bars），可能影响统计显著性评估。

## Further Thoughts

TinyAlign 的检索增强输入信息含量的思路启发性很强，不仅适用于轻量化 VLM，也可扩展至其他资源受限多模态任务；未来可探索动态调整内存库内容或结合外部知识库（如常识库）以提升复杂推理任务的泛化能力；此外，有效互信息（EMI）概念为分析模态对齐提供了新视角，可进一步研究如何量化不同模态间的 EMI 并设计更精确优化目标。