---
title: "Memorization and Knowledge Injection in Gated LLMs"
pubDatetime: 2025-04-30T00:28:32+00:00
slug: "2025-04-gated-llm-memory"
type: "arxiv"
id: "2504.21239"
score: 0.711226834287881
author: "grok-3-latest"
authors: ["Xu Pan", "Ely Hahami", "Zechen Zhang", "Haim Sompolinsky"]
tags: ["LLM", "Continual Learning", "Memory Embedding", "Gating Mechanism", "Knowledge Integration"]
institution: ["Harvard University", "Center for Brain Science at Harvard University", "Kempner Institute at Harvard University", "Edmond and Lily Safra Center for Brain Sciences at Hebrew University"]
description: "本文提出MEGa框架，通过门控LoRA模块将新记忆嵌入大型语言模型权重中，有效缓解持续学习中的灾难性遗忘，并在记忆回忆与知识整合任务上取得显著成果。"
---

> **Summary:** 本文提出MEGa框架，通过门控LoRA模块将新记忆嵌入大型语言模型权重中，有效缓解持续学习中的灾难性遗忘，并在记忆回忆与知识整合任务上取得显著成果。 

> **Keywords:** LLM, Continual Learning, Memory Embedding, Gating Mechanism, Knowledge Integration

**Authors:** Xu Pan, Ely Hahami, Zechen Zhang, Haim Sompolinsky

**Institution(s):** Harvard University, Center for Brain Science at Harvard University, Kempner Institute at Harvard University, Edmond and Lily Safra Center for Brain Sciences at Hebrew University


## Problem Background

大型语言模型（LLMs）在持续学习新记忆和注入新知识时面临灾难性遗忘（Catastrophic Forgetting）问题，无法像人类一样通过逐步经验积累长期记忆（如情景记忆和语义记忆），现有方法（如大上下文窗口或外部数据库检索）也难以实现记忆与语义知识的深度整合。
本文旨在开发一种持续学习框架，使LLMs能够顺序存储新记忆并有效检索，同时保持通用语言能力，探索其作为人类记忆模型的潜力。

## Method

*   **核心思想:** 提出MEGa（Memory Embedded in Gated LLMs）框架，通过门控机制将新记忆直接嵌入模型权重中，避免灾难性遗忘，同时实现记忆的高效检索和知识整合。
*   **记忆编码与存储:** 每个新记忆（以文本形式输入）通过模型内部嵌入生成上下文键（Context Key），并为其初始化一组低秩适应（LoRA）适配器，作为专用权重存储记忆内容。
*   **微调过程:** 针对每个记忆单独微调对应的LoRA适配器，确保记忆嵌入到权重中而不影响模型其他部分，微调时使用特定提示（如‘Tell me a story that you memorized’）以增强记忆与语义的关联。
*   **推理与门控机制:** 在推理时，根据查询（Query）嵌入与存储记忆嵌入的相似性，计算门控权重（Gating Weights），动态组合相关LoRA适配器生成响应，门控权重通过softmax函数调整，确保激活最相关的记忆。
*   **内部RAG（iRAG）策略:** 提出一种内部检索增强生成方法，先通过回忆提示重建相关记忆，再基于回忆内容回答问题，模拟人类记忆激活与工作记忆结合的过程。
*   **关键优势:** 不依赖外部数据库，记忆存储于模型内部权重，模块化设计减少记忆间干扰，同时保留模型通用能力，接近生物学上人类记忆系统的运作方式。

## Experiment

*   **有效性:** MEGa在虚构人物和维基百科2024事件数据集上的记忆回忆任务中，余弦相似度分别达到0.901和0.921，显著优于其他持续学习基线（如LoRA为0.485和0.243）；在问答（QA）任务中准确率分别为72.53%和78.03%，接近RAG（82.57%和88.83%），远超其他方法。
*   **内部RAG提升:** 使用iRAG策略后，QA准确率进一步提升至80.67%和84.70%，几乎与RAG持平，验证了内部回忆机制在知识整合上的潜力。
*   **通用能力保留:** 在MMLU数据集上，MEGa准确率（61.75%和61.99%）接近基础模型（62.56%），而其他方法明显下降（如LoRA降至47.94%和46.88%），表明MEGa对原有知识的保护效果显著。
*   **组合知识任务:** 在需要结合多个记忆的组合问答任务中，MEGa准确率（49.6%和70.4%）接近批量学习（54.4%和75.2%），远超其他持续学习方法，显示门控机制在知识整合上的优势。
*   **实验设置合理性:** 实验涵盖两种数据集（虚构人物模拟情景记忆，维基事件模拟语义记忆），通过多指标（余弦相似度、QA准确率、MMLU准确率）评估模型能力，数据分区（20个分区）和标准差报告增加结果可靠性；但样本规模较小（每分区50个样本），可能限制对大规模记忆存储能力的验证。

## Further Thoughts

MEGa的门控机制和模块化记忆存储启发了对人类互补记忆系统的模拟，未来可探索通过定期‘排练’将LoRA权重蒸馏到基础模型，模拟记忆巩固过程；此外，记忆模块是否可以通过图结构或语义聚类动态合并/拆分，以减少冗余并提升检索效率？若扩展到多模态数据（如图像、音频），是否能更好地模拟情景记忆？参数量线性增长问题是否可通过共享权重或稀疏激活解决？