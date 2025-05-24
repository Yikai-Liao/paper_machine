---
title: "Don't "Overthink" Passage Reranking: Is Reasoning Truly Necessary?"
pubDatetime: 2025-05-22T16:41:37+00:00
slug: "2025-05-reasoning-reranking-analysis"
type: "arxiv"
id: "2505.16886"
score: 0.8066425947926584
author: "grok-3-latest"
authors: ["Nour Jedidi", "Yung-Sung Chuang", "James Glass", "Jimmy Lin"]
tags: ["LLM", "Information Retrieval", "Reranking", "Reasoning", "Partial Relevance"]
institution: ["MIT Lincoln Laboratory", "Massachusetts Institute of Technology", "University of Waterloo"]
description: "本文通过对比实验揭示推理过程在点式重排序中反而损害准确性，因其限制了模型对部分相关性的建模能力，建议使用更简单高效的标准重排序方法。"
---

> **Summary:** 本文通过对比实验揭示推理过程在点式重排序中反而损害准确性，因其限制了模型对部分相关性的建模能力，建议使用更简单高效的标准重排序方法。 

> **Keywords:** LLM, Information Retrieval, Reranking, Reasoning, Partial Relevance

**Authors:** Nour Jedidi, Yung-Sung Chuang, James Glass, Jimmy Lin

**Institution(s):** MIT Lincoln Laboratory, Massachusetts Institute of Technology, University of Waterloo


## Problem Background

随着推理模型在复杂自然语言任务中的成功，信息检索（IR）领域开始探索将显式推理过程（Chain-of-Thought, CoT）引入基于大型语言模型（LLM）的段落重排序器（Passage Reranker），以提升检索精度。然而，显式推理是否真正能改善重排序效果仍未明确。本文研究的核心问题是：推理过程是否能提升点式重排序（Pointwise Reranking）的准确性？研究发现，推理过程可能导致模型输出极化的相关性分数（Polarized Relevance Scores），从而忽略部分相关性（Partial Relevance），这对重排序准确性至关重要。

## Method

* **核心思想：** 对比基于推理的点式重排序器（ReasonRR）和不含推理的标准点式重排序器（StandardRR），以评估推理过程对重排序准确性的影响。
* **具体实现：** 设计三种重排序器变体：
  * **StandardRR：** 标准点式重排序器，直接基于查询-段落对，通过 LLM 微调输出相关性标签（‘true’或‘false’），使用 softmax 计算相关性分数（Relevance Score），不生成推理过程。
  * **ReasonRR：** 基于推理的点式重排序器，在输出相关性标签前，生成显式推理过程（Reasoning Chain），推理内容作为输入的一部分影响最终相关性分数。
  * **ReasonRR-NoReason：** 在推理时禁用推理过程的 ReasonRR 变体，通过预填充一个固定推理过程（如‘<think> Okay, I think I have finished thinking. </think>’），将其转化为标准点式重排序器，输出相关性标签。
* **辅助方法：** 引入自一致性（Self-Consistency）方法，通过多次采样推理过程并平均相关性分数，试图缓解推理过程导致的分数极化问题。
* **训练与评估：** 三种模型在相同训练数据（如 MS MARCO 增强数据集）和骨干 LLM（如 Qwen2.5 系列，1.5B-7B 参数规模）下进行训练和评估，确保控制变量以隔离推理过程的影响。

## Experiment

* **有效性：** 在领域内（MS MARCO）和领域外（BRIGHT）数据集上，StandardRR 普遍优于 ReasonRR，在 MS MARCO 上平均高出 3.7-5.3 NDCG@10 分，在 BRIGHT 上高出 1-3.4 分；ReasonRR-NoReason 比 ReasonRR 更有效，在 MS MARCO 上提升 0.5-1.4 分，在 BRIGHT 上 7B 模型规模下提升 3 分。
* **原因分析：** 推理过程导致 ReasonRR 的相关性分数分布极化，几乎不分配部分相关性分数（0.1-0.9 区间），而 StandardRR 和 ReasonRR-NoReason 能更好地捕捉部分相关性，这对重排序准确性至关重要。
* **改进尝试：** 自一致性方法（ReasonRR + Self-Consistency）通过多次采样推理过程并平均分数，改善了 ReasonRR 的性能（MS MARCO 上提升 1.8 分，BRIGHT 上提升 2.9 分），但仍不及 StandardRR。
* **实验设置：** 实验覆盖不同模型规模（1.5B, 3B, 7B）和多种数据集（MS MARCO 系列和 BRIGHT），设置较为全面合理；但局限性在于未测试更大规模模型（>7B）和其他重排序方法（如 Listwise 或 Setwise）。

## Further Thoughts

推理过程并非所有任务的‘万能钥匙’，在重排序任务中可能引入不必要的复杂性，反而损害性能，这提示 LLM 的推理能力需根据任务特性定制化设计，而非盲目应用；此外，部分相关性的建模对 IR 任务至关重要，未来可探索在推理过程中引入非二元相关性分数（如分级分数 1-5），或设计专门的损失函数校准分数分布，以提升推理重排序器的效果。