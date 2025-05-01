---
title: "XBreaking: Explainable Artificial Intelligence for Jailbreaking LLMs"
pubDatetime: 2025-05-01T18:25:17+08:00
slug: "2025-04-xbbreaking-jailbreaking"
score: 0.7894792847562419
author: "grok-3-mini-latest"
authors: ["Marco Arazzi", "Antonino Nocera", "Vignesh Kumar Kembu", "Vinod P."]
tags: ["LLM", "Explainable AI", "Jailbreaking", "Sampling", "Reasoning"]
institution: ["University of Pavia, Italy", "Cochin University of Science & Technology, India"]
description: "本文提出XBreaking方法，利用Explainable AI分析审查和非审查LLMs的内部模式，识别关键层并通过噪声注入绕过安全机制，同时保持模型功能。"
---

> **Summary:** 本文提出XBreaking方法，利用Explainable AI分析审查和非审查LLMs的内部模式，识别关键层并通过噪声注入绕过安全机制，同时保持模型功能。 

> **Keywords:** LLM, Explainable AI, Jailbreaking, Sampling, Reasoning

> **Recommendation Score:** 0.7894792847562419

**Authors:** Marco Arazzi, Antonino Nocera, Vignesh Kumar Kembu, Vinod P.
**Institution(s):** University of Pavia, Italy, Cochin University of Science & Technology, India

## Problem Background

**论文的出发点和关键问题：** 这篇论文关注大型语言模型（LLMs）的安全性和隐私问题，特别是这些模型在关键应用场景（如政府机构和医疗组织）中的可靠性。LLMs 虽然强大，但其训练数据庞大且可能包含敏感信息，导致数据中毒、泄露和有害内容生成的风险。现有LLMs 通过审查机制（如RLHF 和监督学习）来过滤有害输出，但攻击者可以通过Jailbreaking 绕过这些机制，诱导模型产生不道德或危险的内容。论文指出，现有的Jailbreaking 方法多采用生成和测试策略（如提示工程），缺乏针对性分析，因此无法深入理解审查机制的弱点。论文的核心问题是如何利用Explainable AI（XAI）比较审查模型和非审查模型的行为，识别可利用的模式，从而设计更精确的攻击策略，以揭示和改进LLMs 的安全漏洞。

## Method

**核心思想：** 论文提出XBreaking方法，这是一种基于Explainable AI（XAI）的Jailbreaking攻击策略，旨在通过分析审查模型（Mc）和非审查模型（Mu）的内部表示，识别关键层并注入噪声来破坏安全机制，而不需全面微调模型。

**具体实现步骤：**
- **阶段1：内部表示分析（Internal Representation Profiling）**：假设攻击者有白盒访问权限，对Mc和Mu进行XAI分析（如计算激活值和注意力分数）。具体来说，对于每个层li，计算平均激活分数（使用公式：act_mean = Σ(激活值) / (序列长度 × 隐藏维度)）和平均注意力分数（使用公式：att_mean = Σ(注意力值) / (注意力头数 × 序列长度)），然后进行最小-最大归一化，以识别审查行为相关的层。
- **阶段2：层级区分（Layer Discrimination）**：将层级特征（激活和注意力向量）用于二元分类任务（区分Mc和Mu），采用SelectKBest特征选择方法（如基于卡方测试）来筛选最差异化的层（Top-K层），并通过肘部法（Elbow Method）确定最佳K值。
- **阶段3：噪声注入（Injecting Noise）**：在选定的关键层或其前一层注入Gaussian噪声（如缩放因子为0.1、0.2、0.3），具体方式包括：直接添加到自注意力机制的查询投影权重矩阵，或添加到前一层的层归一化权重向量中。这种方法允许精确破坏审查机制，同时尽量保留模型的整体功能。

**关键特点：** 该方法不修改模型整体参数，仅针对特定层进行手术式干预，降低了计算开销，并通过实验验证了其有效性。

## Experiment

**实验设置：** 论文使用JBB-Behaviors数据集（包含100个有害行为和对应良性行为），在LLaMA 3.2 (1B和3B版本)、Qwen2.5-3B和Mistral-7B-v0.3模型上进行测试。实验包括XAI分析、层选择和噪声注入，使用LLM-as-Judge（Atla Selene Mini模型）评估响应，包括Relevancy、Harmfulness和Hallucination分数（1-5分量表）。还进行了人工验证以确认Judge LLM的准确性（80%与人工一致）。

**实验效果：** 方法提升明显：在保持模型Relevancy的同时，Harmfulness分数显著增加，例如LLaMA 3.2-1B的平均Harmfulness分数从2.04提升至3.21（增加约38%），LLaMA 3.2-3B增加约10%。注入噪声到前一层比直接注入更有效（如Qwen2.5-3B的Harmfulness增加58%）。实验设置全面合理，包括对比基线模型、不同噪声水平测试和类别分析（e.g., Disinformation、Physical Harm），证明了攻击的针对性和泛化性。Hallucination分数在某些模型中略有增加，但Mistral模型显示出较强鲁棒性。总体上，实验数据支持了方法的有效性，并回答了研究问题（RQ1-3），如XAI能可靠区分模型并移除限制。

## Further Thoughts

论文展示了XAI可以精确识别LLMs中负责审查的关键层，这启发未来研究可能将XAI应用于其他模型架构来增强鲁棒性，或开发动态噪声注入机制以平衡安全性和性能；此外，注入噪声的层级策略提示了在不影响整体模型功能的情况下针对性破坏对齐机制的可能性。