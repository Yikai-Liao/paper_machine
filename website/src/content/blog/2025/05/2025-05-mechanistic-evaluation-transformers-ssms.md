---
title: "Mechanistic evaluation of Transformers and state space models"
pubDatetime: 2025-05-21T04:56:09+00:00
slug: "2025-05-mechanistic-evaluation-transformers-ssms"
type: "arxiv"
id: "2505.15105"
score: 0.5914309635234724
author: "grok-3-latest"
authors: ["Aryaman Arora", "Neil Rathi", "Nikil Roashan Selvam", "Róbert Csórdas", "Dan Jurafsky", "Christopher Potts"]
tags: ["Language Models", "Contextual Retrieval", "Mechanistic Interpretability", "Attention Mechanism", "State Space Models"]
institution: ["Stanford University"]
description: "本文提出机制评估框架，通过因果干预揭示 Transformer 和状态空间模型在上下文检索任务上的内在机制差异，为架构设计和解释性研究提供了新工具。"
---

> **Summary:** 本文提出机制评估框架，通过因果干预揭示 Transformer 和状态空间模型在上下文检索任务上的内在机制差异，为架构设计和解释性研究提供了新工具。 

> **Keywords:** Language Models, Contextual Retrieval, Mechanistic Interpretability, Attention Mechanism, State Space Models

**Authors:** Aryaman Arora, Neil Rathi, Nikil Roashan Selvam, Róbert Csórdas, Dan Jurafsky, Christopher Potts

**Institution(s):** Stanford University


## Problem Background

论文关注语言模型架构在上下文检索任务上的表现差异，特别是 Transformer 和状态空间模型（State Space Models, SSMs）在合成任务如 Associative Recall (AR) 上的能力缺陷。
尽管 SSMs 在计算效率和基准测试中表现出色，但它们在从上下文中提取信息的能力上不如 Transformer，传统的行为指标（如准确率）无法解释这种差异的根本原因，因此需要从机制层面理解不同架构在处理上下文检索时的内在工作原理。

## Method

*   **核心思想:** 提出一种机制解释（mechanistic interpretability）框架，通过因果干预分析模型内部信息流动，揭示不同架构在上下文检索任务上的工作机制。
*   **合成任务设计:** 使用 Associative Recall (AR) 任务测试简单的键值对检索能力，并创新提出 Associative Treecall (ATR) 任务，通过基于概率上下文无关语法（PCFG）的树状结构，模拟自然语言中的层次检索需求，增加任务复杂性。
*   **因果干预技术:** 采用 interchange interventions 方法，在模型前向传播中替换特定组件的中间表示（如层、序列混合器等），观察干预是否能恢复正确答案的预测概率，从而判断该组件对任务成功的重要性，分析信息在 token 位置和模型组件间的流动路径。
*   **架构对比实验:** 对比多种架构，包括 Transformer (Attention)、Based、BaseConv、Mamba、H3 和 Hyena，分析它们在 AR 和 ATR 任务上的行为表现和机制差异，识别如归纳头（induction heads）和直接检索（direct retrieval）等机制。
*   **消融研究:** 针对 Mamba 和 Based 模型中的短卷积（short convolutions）组件进行消融实验，通过调整卷积核大小或移除组件，验证其对上下文关联任务表现的影响。

## Experiment

*   **有效性:** 在 AR 任务中，Transformer 和 Based 模型准确率接近 100%，显著优于大多数 SSMs，Mamba 以 91.25% 紧随其后；在 ATR 任务中，Mamba 在小词汇量设置下表现出色（准确率 92.19%），甚至超过 Transformer（80.94%），表明不同架构在层次任务上的潜力。
*   **机制洞察:** 机制分析显示 Transformer 和 Based 通过归纳头机制在值 token 处存储键值关联，而大多数 SSMs 采用直接检索机制，仅在查询 token 或最后状态计算关联；Mamba 的成功部分归因于短卷积组件。
*   **泛化能力:** ATR 任务的训练-测试分割实验表明，Mamba 具有一定泛化能力（测试准确率 65.00%），但不如 Transformer（68.12%）；机制一致性表明模型在分布内和分布外数据上采用相同策略。
*   **消融结果:** 短卷积对 Mamba 和 Based 的表现至关重要，移除或减小卷积核大小导致性能显著下降，例如 Mamba 在 AR 任务上的准确率从 96.25% 降至接近失败。
*   **实验设置合理性:** 实验覆盖多种模型维度（16-256）、学习率范围、任务难度参数（如 ATR 的生产规则长度和词汇量），并通过大量 GPU 小时（<10,000）支持广泛训练和评估；行为指标与机制指标结合，确保分析全面且数据可靠。

## Further Thoughts

机制评估可以作为架构设计的指导工具，通过识别关键组件（如短卷积）的作用，针对特定任务优化模型设计；此外，ATR 任务的树状结构设计启发我们进一步探索多跳推理任务（如祖父母关系查询），以测试模型在复杂上下文中的机制能力；Transformer 的位置无关归纳机制也提示未来可以在 SSMs 中引入类似机制，提升其处理非顺序依赖结构的能力。