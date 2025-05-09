---
title: "Avoid Recommending Out-of-Domain Items: Constrained Generative Recommendation with LLMs"
pubDatetime: 2025-05-06T09:08:36+00:00
slug: "2025-05-constrained-generative-recommendation"
type: "arxiv"
id: "2505.03336"
score: 0.5496516192017691
author: "grok-3-latest"
authors: ["Hao Liao", "Wensheng Lu", "Jianxun Lian", "Mingqi Wu", "Shuo Wang", "Yong Zhang", "Yitian Huang", "Mingyang Zhou", "Xing Xie"]
tags: ["LLM", "Generative Recommendation", "Constrained Generation", "Retrieval", "Domain Adaptation"]
institution: ["Shenzhen University, China", "Microsoft Research Asia", "Microsoft Gaming"]
description: "本文提出 RecLM-cgen 方法，通过约束生成技术消除大型语言模型在推荐系统中的领域外物品推荐问题，同时显著提升推荐准确性，为生成式推荐提供轻量级实用解决方案。"
---

> **Summary:** 本文提出 RecLM-cgen 方法，通过约束生成技术消除大型语言模型在推荐系统中的领域外物品推荐问题，同时显著提升推荐准确性，为生成式推荐提供轻量级实用解决方案。 

> **Keywords:** LLM, Generative Recommendation, Constrained Generation, Retrieval, Domain Adaptation

**Authors:** Hao Liao, Wensheng Lu, Jianxun Lian, Mingqi Wu, Shuo Wang, Yong Zhang, Yitian Huang, Mingyang Zhou, Xing Xie

**Institution(s):** Shenzhen University, China, Microsoft Research Asia, Microsoft Gaming


## Problem Background

大型语言模型（LLMs）在生成式推荐系统中展现出强大的交互能力和潜力，如文本内容融合、冷启动推荐和详细解释，但其生成不在当前领域内物品（Out-of-Domain, OOD）的缺陷可能导致信任问题和负面商业影响。
本文旨在解决这一关键问题，确保推荐物品始终在预定义领域内，从而提升 LLM 推荐系统的可靠性和实用性。

## Method

*   **核心思想:** 通过引入特殊标记和两种不同范式（检索和约束生成），确保 LLM 在推荐物品时不生成领域外内容，同时尽量维持推荐准确性和模型通用能力。
*   **具体方法 1 - RecLM-ret（基于检索）:** 
    *   在模型生成特殊标记 <SOI>（表示物品标题开始）时，提取当前上下文的隐藏层表示，利用嵌入相似性从领域数据集的预计算嵌入中检索最相似的物品标题，随后拼接 <EOI>（物品结束标记）完成推荐。
    *   训练时，使用检索任务损失优化嵌入匹配，并通过投影层对齐模型隐藏状态与物品嵌入空间。
*   **具体方法 2 - RecLM-cgen（基于约束生成）:** 
    *   基于领域内物品标题构建前缀树（Prefix Tree），当模型生成 <SOI> 标记后，限制其解码空间，仅允许生成前缀树中存在的标题，直到生成 <EOI> 标记后恢复正常生成。
    *   引入范围掩码损失（Scope Mask Loss），在训练时限制物品标题相关 token 的 softmax 范围与前缀树一致，提升训练-推理一致性。
    *   加入多轮对话数据（Multi-Round Conversation Data），防止模型在领域任务中丧失通用能力。
*   **实现细节:** 两种方法均基于 Llama3-8B 模型微调，RecLM-cgen 作为轻量级插件模块，仅需少量代码即可集成到现有 LLMs 中，推理时通过自定义 LogitsProcessor 实现约束生成。

## Experiment

*   **有效性:** RecLM-cgen 和 RecLM-ret 在三个数据集（Steam, Movies, Toys）上均将 OOD@10 降至 0%，完全消除了领域外推荐问题，而其他基线（如 GPT-4, Llama3）存在较高 OOD 比例（最高达 90.99%）。
*   **准确性提升:** RecLM-cgen 在推荐准确性上显著优于 RecLM-ret 及其他 LLM 基线，例如在 Steam 数据集上 NDCG@10 提升 6.1%（0.0433 vs. 0.0397），在 Toys 数据集上提升 16.3%（0.0479 vs. 0.0378）；相比传统模型（如 SASRec），RecLM-cgen 在两个数据集上表现更优。
*   **实验设置合理性:** 实验覆盖多个领域（游戏、电影、玩具），采用留一法（Leave-One-Out）数据划分，对比传统推荐模型、通用 LLMs 和微调 LLMs，指标包括准确性（HR@K, NDCG@K）、重复率（Repeat@K）和 OOD 比例（OOD@K）；消融研究验证了约束生成、范围掩码损失和多轮对话数据的贡献。
*   **不足之处:** 实验未深入探讨推荐多样性和公平性，推理延迟问题未解决（工业场景需毫秒级响应），跨领域零样本推荐性能下降明显，需进一步优化。

## Further Thoughts

RecLM-cgen 的约束生成方法（通过前缀树限制解码空间）不仅适用于推荐系统，还可扩展至其他需要结构化输出的任务（如 API 调用、JSON 生成），启发我们在生成任务中探索更紧密的模型内约束机制；此外，单阶段生成优于两阶段检索的结论提示我们关注上下文一致性，而多轮对话数据维持通用能力的策略值得在其他领域（如医疗、教育）中进一步验证。