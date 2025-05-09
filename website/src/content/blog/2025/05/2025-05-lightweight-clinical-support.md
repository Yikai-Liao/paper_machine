---
title: "Lightweight Clinical Decision Support System using QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation"
pubDatetime: 2025-05-06T10:31:54+00:00
slug: "2025-05-lightweight-clinical-support"
type: "arxiv"
id: "2505.03406"
score: 0.5218370905834891
author: "grok-3-latest"
authors: ["Mohammad Shoaib Ansari", "Mohd Sohail Ali Khan", "Shubham Revankar", "Aditya Varma", "Anil S. Mokhade"]
tags: ["LLM", "Retrieval Augmented Generation", "Fine-Tuning", "Quantization", "Clinical Decision Support"]
institution: ["Visvesvaraya National Institute of Technology (VNIT), Nagpur"]
description: "本文提出了一种轻量化的临床决策支持系统，通过结合检索增强生成（RAG）和量化低秩适应（QLoRA）微调技术，提升了大型语言模型在医疗任务中的准确性和效率，同时降低了计算资源需求。"
---

> **Summary:** 本文提出了一种轻量化的临床决策支持系统，通过结合检索增强生成（RAG）和量化低秩适应（QLoRA）微调技术，提升了大型语言模型在医疗任务中的准确性和效率，同时降低了计算资源需求。 

> **Keywords:** LLM, Retrieval Augmented Generation, Fine-Tuning, Quantization, Clinical Decision Support

**Authors:** Mohammad Shoaib Ansari, Mohd Sohail Ali Khan, Shubham Revankar, Aditya Varma, Anil S. Mokhade

**Institution(s):** Visvesvaraya National Institute of Technology (VNIT), Nagpur


## Problem Background

医疗行业面临信息管理和处理的巨大挑战，临床医生需要在不断扩展的医学知识库中提供高质量的患者护理。
通用大型语言模型（LLMs）虽具备处理海量信息的能力，但缺乏特定领域的知识和上下文理解，无法直接应用于高风险的医疗场景。
此外，高计算成本限制了其在资源有限的医疗机构中的部署。

## Method

*   **核心思想:** 构建一个轻量化的临床决策支持系统，通过结合检索增强生成（RAG）和量化低秩适应（QLoRA）微调技术，提升 LLMs 在医疗任务中的准确性和效率，同时降低计算资源需求。
*   **RAG 组件:** 
    *   医院数据（如临床指南、电子健康记录）经过预处理和分割后，使用医疗领域特定的嵌入模型（E5-large-v2）生成向量表示，并存储在向量数据库（如 Pinecone）中。
    *   在运行时，用户查询被嵌入为向量，通过余弦相似度检索最相关的上下文信息，结合系统指令和查询构建提示，输入 LLM 生成响应。
    *   采用混合检索策略（向量相似性搜索+BM25词法搜索）和分层检索机制，确保检索精度和效率。
*   **QLoRA 微调:** 
    *   基于 Llama 3.2-3B-Instruct 模型，利用低秩适应（LoRA）技术，仅更新基础模型权重矩阵的低秩分解部分， trainable 参数占总参数的 0.75%（约 240 万个）。
    *   结合 4-bit 量化技术，将权重从 16-bit 或 32-bit 浮点数压缩为 4-bit 整数，显著降低内存需求（从每 1GB 模型约 2GB VRAM 降至 0.5GB VRAM），并通过双重量化和分页优化器确保训练稳定性。
    *   使用医疗问答数据集（Medical Meadow WikiDoc 和 MedQuAD，共 26,412 个问答对）进行微调，训练 1 个 epoch，损失值为 1.2734。
*   **系统集成:** 将 RAG 检索的上下文与微调后的 LLM 结合，通过精心设计的提示模板，确保响应符合医院协议和医疗知识，同时提供来源归属和置信度指标。

## Experiment

*   **有效性:** QLoRA 微调后的模型在多个医疗基准数据集上表现出显著提升，例如在 MedMCQA 数据集上准确率从 50.9% 提升至 56.39%，在 MMLU Clinical Knowledge 上从 62.64% 提升至 65.28%。
*   **对比分析:** 相较于未微调的 Llama 3.2-3B-Instruct 模型，微调模型在大多数医疗任务中表现更优，但部分子集（如 MMLU College Medicine）略有下降，可能与数据集分布或微调数据量有关。
*   **实验设置:** 实验在 NVIDIA TITAN RTX GPU（24GB VRAM）上进行，训练时间为 5718 秒，吞吐量为 4.619 样本/秒，设置较为全面，涵盖了多任务评估（问答、选择题）。
*   **局限性:** 实验缺乏真实临床环境中的长期测试数据，未充分探讨模型在不同资源环境下的表现差异，与更大规模模型相比仍有性能差距。

## Further Thoughts

QLoRA 的轻量化微调技术不仅适用于医疗领域，还可能推广到其他高专业性领域（如法律、教育），是否可以开发一个通用的轻量化微调框架，适应不同领域需求？
RAG 的动态更新机制启发我思考是否可以通过自动化增量学习或结合实时数据流（如患者监护数据）进一步提升系统适应性。
此外，论文对隐私和伦理的关注提示我们可以在系统设计中嵌入联邦学习或可解释性工具（如注意力可视化），以增强用户信任和系统透明度。