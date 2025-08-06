---
title: "Collaborative Chain-of-Agents for Parametric-Retrieved Knowledge Synergy"
pubDatetime: 2025-08-03T10:00:38+00:00
slug: "2025-08-cocoa-knowledge-synergy"
type: "arxiv"
id: "2508.01696"
score: 0.6062492785689234
author: "grok-3-latest"
authors: ["Yi Jiang", "Sendong Zhao", "Jianbo Li", "Haochun Wang", "Lizhe Zhang", "Yan Liu", "Bing Qin"]
tags: ["LLM", "Retrieval-Augmented Generation", "Knowledge Integration", "Multi-Agent System", "Chain-of-Thought"]
institution: ["Harbin Institute of Technology", "China Mobile Group Heilongjiang Co., Ltd"]
description: "本文提出 CoCoA 框架，通过多代理协作和长链训练策略，显式整合大型语言模型的内部参数化知识和外部检索知识，显著提升了知识密集型任务中的生成性能。"
---

> **Summary:** 本文提出 CoCoA 框架，通过多代理协作和长链训练策略，显式整合大型语言模型的内部参数化知识和外部检索知识，显著提升了知识密集型任务中的生成性能。 

> **Keywords:** LLM, Retrieval-Augmented Generation, Knowledge Integration, Multi-Agent System, Chain-of-Thought

**Authors:** Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, Lizhe Zhang, Yan Liu, Bing Qin

**Institution(s):** Harbin Institute of Technology, China Mobile Group Heilongjiang Co., Ltd


## Problem Background

大型语言模型（LLM）在知识密集型任务中依赖参数化知识，但这种知识难以更新，而检索增强生成（RAG）框架通过引入外部知识提升了性能；然而，当前 RAG 方法在整合模型内部参数化知识（parametric knowledge）和外部检索知识（retrieved knowledge）时存在局限，两种知识来源的协同效果不足，导致生成质量不稳定或未充分利用内部知识。

## Method

* **核心思想：** 提出 CoCoA（Collaborative Chain-of-Agents）框架，通过多代理协作显式整合参数化知识和检索知识，并通过长链训练策略进一步优化模型性能。
* **CoCoA-zero 框架：** 这是一个多代理 RAG 框架，分为两个阶段：
  * **阶段一 - 知识归纳（Knowledge Induction）：** 设计两个专用代理，内部知识代理（Internal Knowledge Agent）基于模型自身知识生成候选答案和支持性段落，外部知识代理（External Knowledge Agent）基于检索文档生成候选答案和支持性段落；通过条件化归纳（conditional induction）将隐式知识显式化。
  * **阶段二 - 高级决策（High-level Decision Making）：** 决策代理（Decision-Making Agent）基于链式思维（Chain-of-Thought, CoT）对内部和外部知识及候选答案进行推理，验证事实准确性和逻辑一致性，最终生成答案；这种设计增强了知识整合的透明度和鲁棒性。
* **CoCoA 训练策略：** 基于 CoCoA-zero 合成的数据，采用长链训练（long-chain training）方式：
  * **监督微调（Supervised Fine-Tuning, SFT）：** 将多代理协作的中间结果（如内部/外部归纳、推理轨迹、最终答案）拼接为长篇响应，训练模型端到端整合知识。
  * **直接偏好优化（Direct Preference Optimization, DPO）：** 使用 CoCoA-zero 的正样本和单代理零样本的负样本进行对比学习，进一步对齐模型的协作行为。
* **关键点：** 方法不仅优化了检索内容的使用，还通过多代理设计和长链训练深度挖掘模型内部知识，实现了两种知识来源的协同增强。

## Experiment

* **有效性：** CoCoA-zero 在无训练情况下，相比标准 RAG 方法，在所有任务上的平均 EM 和 F1 分别提升了 4.99% 和 4.64%；经过训练的 CoCoA（SFT 和 DPO）进一步提升性能，尤其在 2WikiMultiHopQA 上 EM 和 F1 分别提升了 15.2% 和 15.51%，在多个数据集（如 HotpotQA, WebQuestions, TriviaQA）上达到最优。
* **优越性：** 相比其他无训练方法（如 SURE, Self-RAG），CoCoA-zero 展现出更强的知识整合能力；训练后的 CoCoA 超越了基于推理数据蒸馏的模型（如 DeepSeek-R1-8B），表明其在知识密集型任务上的针对性优势。
* **实验设置合理性：** 实验覆盖开放域和多跳问答任务，数据集选择具有代表性；对比基线全面，包括无检索方法、标准 RAG 及优化方法；消融研究验证了各模块（如内部/外部归纳、推理）的必要性；此外，模型在不同文档数量和模型规模下的鲁棒性测试也较为全面。
* **局限性：** 训练数据规模有限，可能影响泛化能力；长上下文瓶颈导致文档数量过多时优势减弱；token 消耗增加可能限制实际应用。

## Further Thoughts

多代理协作的思路可以扩展到其他领域，例如在对话系统中设计专门的‘记忆代理’提取历史信息，与当前输入结合推理；长链训练的概念提示我们，优化整个推理过程而非仅最终输出，可能对复杂任务（如多步推理、长篇生成）有深远影响；此外，是否可以通过动态调整代理角色或数量，适应不同任务需求，也是一个值得探索的方向。