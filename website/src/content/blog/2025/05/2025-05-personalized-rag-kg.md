---
title: "Personalizing Large Language Models using Retrieval Augmented Generation and Knowledge Graph"
pubDatetime: 2025-05-15T04:01:58+00:00
slug: "2025-05-personalized-rag-kg"
type: "arxiv"
id: "2505.09945"
score: 0.3994826239653241
author: "grok-3-latest"
authors: ["Deeksha Prahlad", "Chanhee Lee", "Dongha Kim", "Hokeun Kim"]
tags: ["LLM", "Retrieval Augmented Generation", "Knowledge Graph", "Personalization", "Privacy"]
institution: ["Arizona State University"]
description: "本文提出了一种结合知识图谱（KG）和检索增强生成（RAG）的方法，显著提升大型语言模型（LLM）在个性化响应生成中的准确性和效率，同时保护用户隐私，为本地化部署提供了可行方案。"
---

> **Summary:** 本文提出了一种结合知识图谱（KG）和检索增强生成（RAG）的方法，显著提升大型语言模型（LLM）在个性化响应生成中的准确性和效率，同时保护用户隐私，为本地化部署提供了可行方案。 

> **Keywords:** LLM, Retrieval Augmented Generation, Knowledge Graph, Personalization, Privacy

**Authors:** Deeksha Prahlad, Chanhee Lee, Dongha Kim, Hokeun Kim

**Institution(s):** Arizona State University


## Problem Background

大型语言模型（LLMs）在生成响应时常因缺乏及时、事实性和个性化的输入数据而产生幻觉（hallucinations），导致输出不准确或无关；
本文旨在通过引入检索增强生成（RAG）和知识图谱（KG），为 LLM 提供结构化、动态更新的个人数据（如日历、对话），以生成更符合用户需求的响应，同时保护隐私，避免将敏感数据发送至云端。

## Method

*   **核心思想:** 利用知识图谱（KG）存储和动态更新个人数据，结合检索增强生成（RAG）技术，为大型语言模型（LLM）提供精准上下文，生成个性化响应。
*   **数据构建:** 将个人数据（如日历、对话）转化为知识图谱形式，使用三元组（source, edge, target）表示实体和关系，通过 SpaCy 库提取关系，Networkx 库进行可视化和分析，确保数据结构化和动态更新。
*   **检索与嵌入:** 采用预训练嵌入模型（sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2）将 KG 和原始数据转化为向量嵌入，存储于 FAISS 向量库中，以便基于相似性高效检索相关上下文。
*   **模型优化:** 选用开源模型 Llama-2-Chat（7B、13B、70B 参数规模），通过 bitsandbytes 量化技术优化内存使用，调整生成参数（如 token 限制、重复惩罚、温度）以提升生成质量。
*   **提示与管道:** 设计提示模板（Prompt Templates）引导模型生成简洁响应，利用 LangChain 的 RetrievalQA 管道实现检索与生成结合，通过相似性搜索获取 top-k 相关文档作为上下文。
*   **本地化部署:** 强调在边缘设备上运行小型模型，保护用户隐私，避免敏感数据上传至云端 LLM 提供商。

## Experiment

*   **有效性:** 实验结果表明，提出的方法在 Llama-2-Chat 模型（7B、13B、70B）上显著优于基线（传统 RAG），ROUGE-1 平均提升 35.15%，ROUGE-2 提升 65.57%，ROUGE-L 提升 35.82%，BLEU-1 提升 61.11%，显示出更高的文本准确性和语义一致性。
*   **效率提升:** 执行时间平均减少 8.931%，表明 KG 作为上下文‘路线图’帮助模型更快定位相关信息，减少冗余计算。
*   **实验设置:** 实验在 Intel i9 处理器和 NVIDIA RTX A6000 GPU 环境下进行，覆盖不同规模模型，数据集通过 ChatGPT-4o 生成日历和对话数据，并设计问题-黄金答案对，确保评估无偏；对比基线合理，充分体现 KG 的增益作用。
*   **局限性:** 实验主要聚焦日历数据，未广泛测试其他个人数据类型，泛化性待验证；执行时间减少幅度较小，在资源受限设备上可能仍面临挑战。

## Further Thoughts

知识图谱（KG）的结构化特性可扩展至更多个人数据领域（如医疗记录、购物偏好），构建多维度用户画像以提升个性化效果；
在边缘设备运行小型 LLM 的思路启发模型压缩和量化技术的进一步优化，以适应低功耗设备；
KG 的动态更新特性提示探索自适应学习机制，根据用户交互实时调整 KG 结构；
结合多模态数据（如图像、语音）构建多模态 KG，可能为 LLM 提供更丰富上下文，适用于复杂交互场景。