---
title: "$\textit{New News}$: System-2 Fine-tuning for Robust Integration of New Knowledge"
pubDatetime: 2025-05-03T12:49:35+00:00
slug: "2025-05-system2-finetuning-news"
type: "arxiv"
id: "2505.01812"
score: 0.751370739897321
author: "grok-3-latest"
authors: ["Core Francisco Park", "Zechen Zhang", "Hidenori Tanaka"]
tags: ["LLM", "Fine-Tuning", "In-Context Learning", "Knowledge Integration", "Data Augmentation"]
institution: ["Harvard University", "CBS-NTT Program in Physics of Intelligence", "Center for Brain Science"]
description: "本文提出 System-2 Fine-tuning（Sys2-FT）方法，通过自我生成数据显著提升大型语言模型对新知识的权重内学习能力，并揭示上下文遮蔽效应对微调的影响。"
---

> **Summary:** 本文提出 System-2 Fine-tuning（Sys2-FT）方法，通过自我生成数据显著提升大型语言模型对新知识的权重内学习能力，并揭示上下文遮蔽效应对微调的影响。 

> **Keywords:** LLM, Fine-Tuning, In-Context Learning, Knowledge Integration, Data Augmentation

**Authors:** Core Francisco Park, Zechen Zhang, Hidenori Tanaka

**Institution(s):** Harvard University, CBS-NTT Program in Physics of Intelligence, Center for Brain Science


## Problem Background

大型语言模型（LLMs）在上下文学习（In-Context Learning, ICL）中能够有效处理新信息（news），但通过微调（Fine-Tuning, FT）将这些知识固化到模型权重中仍面临挑战。
这一问题导致模型难以在动态环境中持续适应新知识，作者旨在缩小微调与上下文学习之间的性能差距（FT-ICL gap），以提升模型的权重内学习能力。

## Method

*   **核心思想:** 提出 System-2 Fine-tuning（Sys2-FT），一种受认知科学中记忆巩固和信息重放启发的微调框架，利用模型自身的上下文学习能力生成合成数据（replay elements），将上下文中的知识转移到权重中。
*   **具体实现:** Sys2-FT 包含三种数据生成协议：
    *   **Paraphrase 协议**：提示模型生成新闻的多种改写版本，通过多样化表达增强数据丰富性，帮助模型从不同角度理解同一信息。
    *   **Implication 协议**：引导模型推理新闻的下游影响或后果，生成分析性内容，旨在帮助模型理解新信息的深层含义和逻辑关联。
    *   **Self-QA 协议**：通过两阶段对话生成问题和答案对，首先让模型基于新闻生成问题，然后在上下文支持下生成准确答案，确保知识的深度内化。
*   **数据格式设计:** 训练数据以对话形式组织，包含用户提问和模型回复，避免直接将新闻作为上下文前缀，以防止上下文遮蔽效应（Contextual Shadowing Effect）。
*   **关键特点:** 不依赖外部知识或复杂损失函数（如 KL 散度），而是通过简单的自我生成数据增强微调效果，同时注重数据多样性和训练稳定性。

## Experiment

*   **有效性:** Sys2-FT 显著优于朴素微调，尤其在 Self-QA 协议下，模型在数学和编程领域的下游问题准确率接近上下文学习（ICL）水平，例如在 Qwen2.5-14B 模型上，数学领域的准确率从朴素微调的低值提升至接近 ICL 的高值。
*   **模型规模影响:** 较大模型（3B以上）在 Sys2-FT 中表现出更高的样本效率，达到相同准确率所需的训练数据更少，显示出规模对新知识整合的重要性。
*   **领域差异:** 量化领域（如数学、编程）从 Sys2-FT 中获益更多，准确率提升明显，而非量化领域（如事件、排行榜）提升有限，可能与任务特性或预训练数据分布有关。
*   **上下文遮蔽效应:** 实验验证了上下文前缀会严重干扰权重内学习，训练数据中包含新闻上下文时，学习效果几乎完全丧失，这一现象在所有模型规模和协议中均一致。
*   **实验设置合理性:** 实验基于 *New News* 数据集，涵盖五个领域、不同模型规模（0.5B至32B），并通过随机化选项顺序等方法减少评估偏差，设置较为全面；但非量化领域提升有限及扩展规律的初步性提示未来研究方向。
*   **计算开销与扩展性:** Sys2-FT 主要增加数据生成阶段的计算成本，但初步发现其存在计算相关的扩展规律（scaling law），大模型在相同计算量下达到相似准确率，显示出样本效率优势。

## Further Thoughts

Sys2-FT 的自我重放机制启发我们可以在持续学习任务中引入类似的数据增强策略，通过模型自生成内容模拟人类记忆巩固过程；上下文遮蔽效应的发现提示在设计微调或预训练数据时，应避免过早暴露关键上下文信息，可能通过分阶段学习或数据掩码等方式优化；此外，领域差异性表明未来可以针对非量化领域引入外部知识源（如知识图谱）或混合协议（如结合 Self-QA 和 Implication），以提升知识内化效果。