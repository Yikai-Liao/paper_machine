---
title: "MARK: Memory Augmented Refinement of Knowledge"
pubDatetime: 2025-05-08T12:28:00+00:00
slug: "2025-05-memory-augmentation-mark"
type: "arxiv"
id: "2505.05177"
score: 0.6686229325448849
author: "grok-3-latest"
authors: ["Anish Ganguli", "Prabal Deb", "Debleena Banerjee"]
tags: ["LLM", "Memory Augmentation", "Domain Adaptation", "Context Retention", "Personalization"]
institution: ["Microsoft Industry Solutions Engineering"]
description: "本文提出 MARK 框架，通过多代理记忆系统增强大型语言模型的领域知识适应能力，利用结构化记忆精炼和注入机制显著提升响应准确性和上下文一致性，无需频繁微调。"
---

> **Summary:** 本文提出 MARK 框架，通过多代理记忆系统增强大型语言模型的领域知识适应能力，利用结构化记忆精炼和注入机制显著提升响应准确性和上下文一致性，无需频繁微调。 

> **Keywords:** LLM, Memory Augmentation, Domain Adaptation, Context Retention, Personalization

**Authors:** Anish Ganguli, Prabal Deb, Debleena Banerjee

**Institution(s):** Microsoft Industry Solutions Engineering


## Problem Background

大型语言模型（LLMs）在专业领域任务中表现出色，但由于依赖预训练知识，难以适应不断演变的领域知识，尤其是在高风险领域（如医疗、金融）中，传统微调成本高昂且不切实际。
此外，现有模型缺乏实时适应能力和跨会话的用户反馈持久化机制，导致重复错误和幻觉（Hallucinations），而检索增强生成（RAG）方法常因格式或上下文不匹配而失效。
MARK 框架旨在通过记忆增强机制，使 LLMs 能够在不依赖频繁微调的情况下持续学习和精炼领域知识。

## Method

*   **核心思想:** 提出 Memory Augmented Refinement of Knowledge (MARK) 框架，通过多代理记忆系统增强 LLMs 的领域适应能力，利用结构化记忆精炼知识并持久化上下文。
*   **架构设计:** MARK 分为两个主要模块：
    *   **记忆构建服务 (Memory Builder Service, MBS):** 从用户与代理的对话历史中提取三种精炼记忆：
        - **残余精炼记忆 (Residual Refined Memory):** 存储领域特定洞察和长期上下文，用于提升模型对领域逻辑的理解。
        - **用户问题精炼记忆 (User Question Refined Memory):** 捕捉用户提供的术语、缩写和事实，确保模型理解用户意图。
        - **LLM 响应精炼记忆 (LLM Response Refined Memory):** 提取响应中的关键元素，用于个性化表达和响应优化。
        这些记忆以文本、向量化和元数据形式存储在支持向量搜索的文档库中。
    *   **记忆搜索服务 (Memory Search Service, MSS):** 在新对话中基于用户问题的相似性评分（Similarity Score）检索相关记忆，并通过记忆相关性评分（Memory Relevance Scoring, MRS）重新排序，MRS 综合考虑回忆次数（Recall Count）、新鲜度（Recency）、相似性评分和反馈评分（Feedback Score），最终将最高评分的记忆注入到 LLM 上下文中。
*   **风险控制:** 引入信任评分（Trust Score）和持久性评分（Persistence Score）机制，动态评估记忆可靠性，防止错误信息传播。
*   **优势:** 不修改原始模型，仅通过记忆注入实现动态适应，支持多轮多用户交互，确保知识精炼的可持续性。

## Experiment

*   **有效性:** 在 MedMCQA 医疗数据集上，MARK 框架显著提升了响应准确性，平均信息捕获评分（AICS）从 0.18 提升至 0.36（100% 提升），关键点覆盖率（KPCS）从 0.12 提升至 0.32（166.7% 提升），错误响应比例下降 67.4%。
*   **效率提升:** 记忆注入后，平均每响应 Token 数从 415 减少至 149，表明系统生成更简洁且信息密集的输出。
*   **实验设置合理性:** 实验分为四阶段（基线评估、记忆构建、记忆注入、多轮多用户交互），覆盖从单轮到复杂交互场景，数据选择针对高风险医疗领域，评估使用 GPT-3.5 构建和 GPT-4 验证，确保结果客观。
*   **局限性:** 实验主要聚焦医疗领域，跨领域泛化能力未充分验证；此外，记忆积累对系统性能的长期影响需进一步研究。

## Further Thoughts

MARK 的多代理记忆系统启发了对 LLM 模块化设计的思考，是否可以为不同领域（如医疗、法律）定制专用代理，进一步提升跨领域适应性？
此外，信任评分机制是否可引入用户角色权重（如专家 vs 普通用户），以更精准过滤错误信息？
最后，是否能将记忆注入与知识图谱结合，通过结构化领域知识表示提升上下文检索和推理能力？