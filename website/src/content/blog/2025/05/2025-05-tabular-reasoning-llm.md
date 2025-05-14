---
title: "How well do LLMs reason over tabular data, really?"
pubDatetime: 2025-05-12T11:35:28+00:00
slug: "2025-05-tabular-reasoning-llm"
type: "arxiv"
id: "2505.07453"
score: 0.7022590012797727
author: "grok-3-latest"
authors: ["Cornelius Wolff", "Madelon Hulsebos"]
tags: ["LLM", "Tabular Data", "Reasoning", "Evaluation Metrics", "Robustness"]
institution: ["Centrum Wiskunde & Informatica"]
description: "本文通过提出 LLM-as-a-judge 评估方法和引入现实世界表格特性，揭示了大型语言模型在表格推理上的显著性能不足，并强调了提升鲁棒性的必要性。"
---

> **Summary:** 本文通过提出 LLM-as-a-judge 评估方法和引入现实世界表格特性，揭示了大型语言模型在表格推理上的显著性能不足，并强调了提升鲁棒性的必要性。 

> **Keywords:** LLM, Tabular Data, Reasoning, Evaluation Metrics, Robustness

**Authors:** Cornelius Wolff, Madelon Hulsebos

**Institution(s):** Centrum Wiskunde & Informatica


## Problem Background

大型语言模型（LLMs）在自然语言任务上表现出色，但其在表格数据上的推理能力尚未被充分验证，尤其是在面对现实世界中常见的表格特性（如缺失值、重复实体、结构变化）时；此外，现有评估方法（如 SacreBleu 和 BERT-score）无法准确反映模型在分析性表格查询上的真实性能，部分基准测试（如 TQA-Bench）通过多选题形式泄露标准答案，影响评估可靠性，因此论文聚焦于探讨 LLMs 在现实表格数据下的推理鲁棒性以及如何更真实地评估其性能。

## Method

* **评估方法改进：** 针对现有评估方法的不足，提出使用 'LLM-as-a-judge' 方法，即利用另一个 LLM 作为评判者，通过结构化提示比较模型生成的开放式答案与标准答案，判断其正确性；该方法避免了多选题泄露答案的问题，并通过与人工标注的对比验证了其可靠性（正确答案识别率 95.8%，错误答案识别率 99.2%）。
* **鲁棒性测试扩展：** 基于 TQA-Bench 基准，对表格数据引入三种现实世界特性：缺失值（随机移除关键单元格并重新计算标准答案）、重复实体（随机复制行以模拟数据冗余）和结构变化（打乱行列顺序以测试结构无关性）；同时，缩减表格数据规模（1K 到 8K 令牌）以获得细粒度性能洞察；评估不仅关注答案准确性，还关注模型是否能承认数据问题（如缺失值）。
* **模型与任务：** 测试了多种 LLMs（如 Qwen2.5、Llama3.1、GPT-4o-mini）在不同复杂度的推理任务（查找、聚合、复杂计算）上的表现，使用统一提示模板引导模型基于表格数据回答问题。

## Experiment

* **评估方法有效性：** 传统指标（如 SacreBleu 和 BERT-score）在区分正确与错误答案时分布重叠，无法提供可靠信号，而 LLM-as-a-judge 方法表现出高准确率，重新评估 TQA-Bench 数据集后发现 LLMs 真实性能远低于多选题评估结果（例如在 '平均值' 和 '减法' 任务上分别低 30% 和 60%）。
* **推理性能与规模：** LLMs 性能随表格数据规模增加而下降，尤其在复杂任务（如相关性计算、减法）上表现较差；GPT-4o-mini 表现最稳定，其他模型（如 Llama3.1）在 4K 以上规模数据上准确率显著下降。
* **鲁棒性测试：** 缺失值和重复实体对准确率影响较大，例如 Llama3.1 在缺失值下的 '求和' 任务准确率从 16% 降至 8%，但部分模型（如 Qwen2.5）在缺失值场景下有所改进；模型能部分承认数据问题（承认率约 44%-60%），但整体不稳定；结构变化影响较小，表明模型对表格结构有一定鲁棒性。
* **实验设置合理性：** 实验涵盖多种模型、任务复杂度、数据规模及现实特性，LLM-as-a-judge 方法提高了评估可信度，但模型行为的不一致性（如推理混乱、代码生成倾向）需进一步定性分析。

## Further Thoughts

LLM-as-a-judge 方法可推广至其他复杂生成任务的评估，如开放式问答或代码生成；通过针对缺失值和重复实体进行数据增强或结合传统数据库查询技术，或许能提升 LLMs 在表格推理中的鲁棒性；此外，训练模型生成带有不确定性声明的答案（如承认数据缺失）可能提高下游任务的可信度。