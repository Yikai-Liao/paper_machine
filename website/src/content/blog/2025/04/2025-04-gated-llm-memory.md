---
title: "Memorization and Knowledge Injection in Gated LLMs"
pubDatetime: 2025-05-01T15:49:20Z
slug: "2025-04-gated-llm-memory"
type: "arxiv"
id: "2504.21239"
score: 0.9081774328496968
author: "grok-3-latest"
authors: ["Xu Pan", "Ely Hahami", "Zechen Zhang", "Haim Sompolinsky"]
tags: ["LLM", "Continual Learning", "Memory Injection", "Gating Mechanism", "Knowledge Integration"]
institution: ["Harvard University", "Center for Brain Science at Harvard University", "Kempner Institute at Harvard University", "Edmond and Lily Safra Center for Brain Sciences at Hebrew University"]
description: "本文提出MEGa框架，通过门控LoRA适配器将新记忆嵌入大型语言模型权重中，有效缓解持续学习中的灾难性遗忘，并在记忆回忆与问答任务中取得显著性能提升。"
---

> **Summary:** 本文提出MEGa框架，通过门控LoRA适配器将新记忆嵌入大型语言模型权重中，有效缓解持续学习中的灾难性遗忘，并在记忆回忆与问答任务中取得显著性能提升。 

> **Keywords:** LLM, Continual Learning, Memory Injection, Gating Mechanism, Knowledge Integration
> **Recommendation Score:** 0.9081774328496968

**Authors:** Xu Pan, Ely Hahami, Zechen Zhang, Haim Sompolinsky
**Institution(s):** Harvard University, Center for Brain Science at Harvard University, Kempner Institute at Harvard University, Edmond and Lily Safra Center for Brain Sciences at Hebrew University

## Problem Background

大型语言模型（LLMs）在持续学习新记忆和注入新知识时面临灾难性遗忘（Catastrophic Forgetting）问题，无法像人类一样通过连续经验积累长期记忆（如情景记忆和语义记忆）。
现有方法（如大上下文窗口或检索增强生成RAG）要么容量有限，要么无法深度整合语义知识，缺乏对日常生活中事件记忆的模拟。

## Method

*   **核心思想:** 提出MEGa（Memory Embedded in Gated LLMs）框架，通过门控机制将新记忆直接嵌入模型权重中，实现持续学习，同时减少灾难性遗忘。
*   **具体实现:** 
    *   **记忆存储:** 为每个新记忆样本初始化一组低秩适应（LoRA）适配器权重，将记忆内容嵌入模型权重中，并存储样本的嵌入作为‘上下文键’（Context Key）。
    *   **微调过程:** 在微调时，使用特定提示（如‘告诉我你记住的故事’）对每个LoRA适配器单独训练，确保记忆内容被有效编码，同时避免干扰其他记忆或模型原有知识。
    *   **推理过程:** 在推理时，通过计算查询嵌入与存储的上下文键之间的语义相似性，生成门控权重（Gating Weights），动态激活与查询最相关的LoRA适配器权重，从而激活对应记忆。
    *   **任务支持:** 支持记忆回忆（Recall）任务，即重建完整记忆内容；支持问答（QA）任务，即基于记忆回答问题；还提出内部RAG（iRAG）方法，先回忆相关记忆再基于回忆内容回答问题。
*   **关键优势:** 不依赖外部数据库，记忆存储于模型内部，更接近人类记忆的生物学机制；通过门控隔离不同记忆，减少干扰；支持组合推理，即通过加权激活多个LoRA适配器回答复杂问题。

## Experiment

*   **有效性:** MEGa在记忆回忆任务中表现出色，余弦相似度在虚构人物数据集上为0.901，在维基百科2024事件数据集上为0.921，远超其他持续学习基线（如LoRA的0.485和0.243）；在问答任务中，准确率分别为72.53%和78.03%，接近RAG（82.57%和88.83%）。
*   **灾难性遗忘缓解:** MEGa的遗忘曲线较为平缓，表明其有效减少了新记忆对旧记忆的干扰，相比之下，其他基线方法在持续学习中性能迅速下降。
*   **通用知识保留:** 在MMLU数据集上，MEGa的准确率（61.75%和61.99%）接近基础模型，未显著损害通用语言能力，而其他基线（如LoRA）下降明显（47.94%和46.88%）。
*   **内部RAG提升:** iRAG方法通过先回忆再回答，将问答准确率提升至80.67%和84.70%，接近RAG性能。
*   **实验设置合理性:** 实验覆盖两个数据集（虚构人物和维基百科事件），评估了回忆、问答和组合问答任务，报告了20个数据集分区的均值和标准差，结果可靠；但样本规模较小（每个分区50个样本），可能未反映大规模场景下的挑战。
*   **局限性:** 参数数量随记忆样本线性增长，增加存储和计算成本；门控选择依赖嵌入质量，正确率分别为85.0%和87.8%，仍有改进空间。

## Further Thoughts

MEGa的门控机制和模块化记忆存储启发了对持续学习的新思考：是否可以通过知识图谱或多层次门控机制提高记忆选择精度和组合推理能力？此外，受人类互补记忆系统启发，是否可以通过模拟记忆巩固过程，将LoRA权重逐步蒸馏到基础模型中，实现参数压缩和长期记忆整合？最后，扩展到多模态记忆时，如何设计跨模态门控机制以关联文本、图像和音频记忆，是一个值得探索的方向。