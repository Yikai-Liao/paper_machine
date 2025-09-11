---
title: "HyFedRAG: A Federated Retrieval-Augmented Generation Framework for Heterogeneous and Privacy-Sensitive Data"
pubDatetime: 2025-09-08T08:44:24+00:00
slug: "2025-09-hyfedrag-federated-rag"
type: "arxiv"
id: "2509.06444"
score: 0.46485791981056734
author: "grok-3-latest"
authors: ["Cheng Qian", "Hainan Zhang", "Yongxin Tong", "Hong-Wei Zheng", "Zhiming Zheng"]
tags: ["LLM", "Federated Learning", "Retrieval-Augmented Generation", "Privacy Preservation", "Heterogeneous Data"]
institution: ["Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing", "Institute of Artificial Intelligence, Beihang University, China"]
description: "HyFedRAG 提出了一种联邦式检索增强生成框架，通过边缘-云协同机制统一处理异构数据并保护隐私，同时利用三级缓存显著提升系统效率，在医疗数据检索中表现出色。"
---

> **Summary:** HyFedRAG 提出了一种联邦式检索增强生成框架，通过边缘-云协同机制统一处理异构数据并保护隐私，同时利用三级缓存显著提升系统效率，在医疗数据检索中表现出色。 

> **Keywords:** LLM, Federated Learning, Retrieval-Augmented Generation, Privacy Preservation, Heterogeneous Data

**Authors:** Cheng Qian, Hainan Zhang, Yongxin Tong, Hong-Wei Zheng, Zhiming Zheng

**Institution(s):** Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing, Institute of Artificial Intelligence, Beihang University, China


## Problem Background

传统集中式检索增强生成（RAG）系统难以处理分布在不同机构、格式各异的异构数据（如结构化 SQL、半结构化知识图谱、非结构化文本），尤其是在医疗等隐私敏感领域，隐私法规（如 GDPR 和 HIPAA）限制了数据的集中化存储和共享，导致跨机构查询（如罕见病案例检索）面临隐私限制和数据格式不一致的挑战。

## Method

* **核心思想**：提出 HyFedRAG，一个联邦式检索增强生成框架，通过边缘-云协同机制，在保护数据隐私的同时统一处理异构数据，实现跨机构的检索和生成。
* **架构设计**：框架分为三层：客户端层负责本地多模态检索和隐私保护摘要生成，中间层提供多级缓存和调度，中央服务器层聚合去标识化的摘要并进行全局推理；基于 Flower 联邦学习框架实现分布式协作。
* **隐私保护机制**：在客户端使用三种工具实现多粒度隐私保护，包括 Presidio 进行基于规则和模型的个人身份信息（PII）掩码，Eraser4RAG 进行上下文相关的非必要信息过滤，以及 TenSEAL 进行同态加密保护特征级数据，确保原始敏感数据不离开本地。
* **多模态检索策略**：针对不同数据模态设计专用检索模块，文本检索结合稀疏（TF-IDF）和稠密（BGE 嵌入）方法并通过混合重排序优化结果，知识图谱检索通过实体匹配和语义重排序提取相关患者路径，SQL 检索结合布尔和自然语言模式并使用深度重排序模型筛选结果。
* **效率优化**：设计三级缓存机制，包括本地摘要特征缓存、摘要到 LLM 输入转换缓存和云端推理缓存，减少冗余计算和通信延迟。

## Experiment

* **检索性能**：在 PMC-Patients 数据集上，HyFedRAG 在文本数据检索中显著优于基线模型（如 DPR、BM25），MRR 达到 39.63%（相对提升 11.87%），nDCG@10 达到 41.33%（相对提升 17.21%）；但在 SQL 和知识图谱数据上性能下降（MRR 分别为 23.01% 和 9.79%），反映出结构化数据语义信息提取的不足。
* **隐私保护效果**：通过 DeepEval 框架（基于 GPT-4o 评估），HyFedRAG 的隐私保护机制显著提升了生成内容的隐私评分，表明其在保护敏感信息的同时保留了文本可读性和信息完整性。
* **系统效率**：三级缓存机制将推理延迟降低了约 80%，缓存命中率超过 84%，有效减少了通信和计算开销。
* **实验设置合理性**：实验覆盖了多种数据模态和基线模型，评估指标（MRR、P@K、nDCG@K）符合信息检索标准，模拟联邦检索设置和缓存命中率分析增强了现实意义；但未对不同隐私工具的效果进行单独消融分析，难以判断各工具的具体贡献。

## Further Thoughts

HyFedRAG 启发了我思考如何进一步优化联邦 RAG 框架，例如通过自适应融合策略动态调整检索模块的权重（如 α 值），根据查询类型或数据模态自动优化性能；此外，针对结构化数据（如 SQL 和知识图谱）检索性能较低的问题，可以探索语义增强方法，如将结构化数据转化为文本描述后再处理；同时，隐私与性能的权衡也值得深入研究，是否可以设计基于用户需求的动态隐私级别调整机制，在低敏感场景下降低加密强度以提升效率。