---
title: "GenKI: Enhancing Open-Domain Question Answering with Knowledge Integration and Controllable Generation in Large Language Models"
pubDatetime: 2025-05-26T08:18:33+00:00
slug: "2025-05-genki-knowledge-integration"
type: "arxiv"
id: "2505.19660"
score: 0.6685998794753515
author: "grok-3-latest"
authors: ["Tingjia Shen", "Ruijun Sun", "Hao Wang", "Yang Song", "Chuan Qin", "Defu Lian", "Hengshu Zhu", "Enhong Chen"]
tags: ["LLM", "Knowledge Integration", "Controllable Generation", "Retrieval", "OpenQA"]
institution: ["University of Science and Technology of China", "BOSS Zhipin Career Science Lab", "Chinese Academy of Sciences"]
description: "本文提出GenKI框架，通过三阶段范式（知识检索、知识整合、可控生成）增强大型语言模型在开放域问答中的性能，有效解决知识缺陷和答案格式对齐问题，并在多个数据集上显著优于基线模型。"
---

> **Summary:** 本文提出GenKI框架，通过三阶段范式（知识检索、知识整合、可控生成）增强大型语言模型在开放域问答中的性能，有效解决知识缺陷和答案格式对齐问题，并在多个数据集上显著优于基线模型。 

> **Keywords:** LLM, Knowledge Integration, Controllable Generation, Retrieval, OpenQA

**Authors:** Tingjia Shen, Ruijun Sun, Hao Wang, Yang Song, Chuan Qin, Defu Lian, Hengshu Zhu, Enhong Chen

**Institution(s):** University of Science and Technology of China, BOSS Zhipin Career Science Lab, Chinese Academy of Sciences


## Problem Background

开放域问答（OpenQA）是自然语言处理（NLP）中的核心任务，旨在从非结构化文本中提取答案。随着大型语言模型（LLMs）的快速发展，其在OpenQA中的应用展现出强大的理解和生成能力，但面临两大关键挑战：一是知识缺陷（Knowledge Deficiency），即LLMs对不常见知识的记忆有限，容易产生幻觉（Hallucination）；二是答案格式对齐（Answer Format Alignment）问题，因不同数据集答案格式各异，LLMs输出常偏离预期。现有方法如预训练或基于提示的知识增强要么计算成本高，要么效果不稳定，因此需要一种新框架同时解决知识整合和可控生成问题。

## Method

*   **核心思想:** 提出GenKI框架，通过三阶段范式（知识检索、知识整合、可控生成）增强LLMs在OpenQA中的性能，解决知识缺陷和答案格式对齐问题。
*   **知识检索（Knowledge Retriever）:** 采用密集段落检索（Dense Passage Retrieval, DPR）模型，从知识库中提取与问题相关的段落，确保输入知识的高质量。
*   **知识整合（Knowledge Integration）:** 创新性地结合自回归训练损失和监督微调损失，将检索到的知识融入LLM参数中，而非仅作为提示输入，确保知识存储更稳定；具体通过LoRA轻量微调方法优化模型，减少计算资源需求，同时设计特定提示（如学习知识和回答问题）辅助训练。
*   **可控生成（Controllable Generation）:** 利用微调后的LLM对生成答案进行后处理，确保格式对齐；引入基于文本一致性的集成方法，结合奖励模型（Reward Model）评估答案连贯性和流畅性，以及外部选择模型（如ChatGPT）确保准确性，通过评分机制（如NISF和长度加权）从多个候选答案中选择最优输出。
*   **关键创新:** 三阶段设计避免了传统两阶段RAG中知识整合与生成任务的分布差异，通过参数存储知识提升稳定性，并通过集成方法实现答案格式的精准控制。

## Experiment

*   **有效性:** 实验在TriviaQA、MSMARCO和CMRC2018三个数据集上进行，GenKI显著优于基线模型。例如，在TriviaQA上，LLaMA-65B结合GenKI后EM指标提升4.8%，F1提升5.6%；在CMRC2018上，GLM-6B的EM指标从1.8%提升至78.3%，效果极为显著。
*   **优越性:** 相比任务特定基线（如ChatGPT、ERNIE-Gram）、最新LLM方法（如Self-RAG、RFiD）和骨干模型（如LLaMA-65B、GLM-6B），GenKI在知识整合和格式控制能力上均表现出色，尤其在跨域和知识库独立性测试中展现了鲁棒性。
*   **实验设置合理性:** 数据集选择涵盖不同答案格式（自由答案、片段答案）和语言（英文、中文），评价指标包括Exact Match（EM）、F1、BLEU、ROUGE和Coherence，全面评估知识整合和可控生成能力；消融实验揭示检索质量与模型知识召回的线性关系，验证了各模块贡献。
*   **局限性:** 在MSMARCO的ROUGE-L指标上稍逊于某些基线，可能因数据集答案偏向长段落，而GenKI更注重知识精炼；此外，计算成本未详细讨论，可能影响实际应用。

## Further Thoughts

GenKI的三阶段范式（检索、知识整合、可控生成）为LLM在知识密集型任务中的应用提供了新思路，启发我们思考是否可以将知识整合通过参数存储的方式推广到其他任务（如对话系统或知识推理），以提升模型对外部知识的稳定利用；此外，检索质量与模型性能的线性关系及瓶颈现象提示我们，未来可以通过改进检索模型或引入多轮检索机制进一步提升效果；可控生成模块的集成方法也启发我们探索更高效的奖励模型设计，以在计算成本和生成质量之间找到平衡。