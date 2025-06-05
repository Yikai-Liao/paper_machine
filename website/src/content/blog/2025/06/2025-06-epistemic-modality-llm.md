---
title: "Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes"
pubDatetime: 2025-06-02T10:19:42+00:00
slug: "2025-06-epistemic-modality-llm"
type: "arxiv"
id: "2506.01512"
score: 0.593537939451183
author: "grok-3-latest"
authors: ["Meng Li", "Michael Vrazitulis", "David Schlangen"]
tags: ["LLM", "Epistemic Modality", "Uncertainty Expression", "Semantic Knowledge", "Theory of Mind"]
institution: ["University of Potsdam"]
description: "本文通过受控故事首次系统评估了大型语言模型在认知情态语义知识上的表现，揭示其生成不确定性表达的局限性，并为构建更理性的模型提供了理论指导。"
---

> **Summary:** 本文通过受控故事首次系统评估了大型语言模型在认知情态语义知识上的表现，揭示其生成不确定性表达的局限性，并为构建更理性的模型提供了理论指导。 

> **Keywords:** LLM, Epistemic Modality, Uncertainty Expression, Semantic Knowledge, Theory of Mind

**Authors:** Meng Li, Michael Vrazitulis, David Schlangen

**Institution(s):** University of Potsdam


## Problem Background

随着大型语言模型（LLMs）在现实世界中的广泛应用，模型需要像理性人类一样，根据证据的可信度和自身信心生成合适的语言表达，尤其是在涉及不确定性和认知态度（epistemic modality）时。
然而，当前模型在生成与不确定性相关的认知表达（epistemic expressions）时是否具备足够的语义知识，以及其表达是否可靠，仍未得到充分考察。
论文指出，现有研究多关注通过提示（prompting）让模型表达不确定性，但忽视了模型是否真正掌握这些语言知识的根本问题。

## Method

*   **核心思想:** 通过行为评估方法，推断大型语言模型在认知情态（epistemic modality）方面的语义知识，而非直接测量其内部表征，重点测试模型在受控情境下生成合适认知表达的能力。
*   **实验设计:** 设计了两组实验，基于语言学和儿童语言发展的研究范式：
    *   **实验 1 - 情态助动词（Modal Auxiliaries）:** 测试模型对可能性（possibility，如 may/might）和必要性（necessity，如 must/have to）的理解，使用 150 个由 5 个模板生成的故事，控制证据类型和情态条件。
    *   **实验 2 - 态度动词（Attitude Verbs）:** 测试模型在不同确定性程度下报告事实和信念的能力，使用 30 个 Theory of Mind (ToM) 故事，聚焦于动词如 know, believe, doubt。
*   **刺激与变量控制:** 故事结合手动编写和模板生成，控制了证据类型、确定性程度、故事类型（base, 1-shot, counterfactual）和提问格式（direct slot, indirect slot, indirect sentence），以减少外部知识干扰。
*   **模型与实现:** 评估了 8 个开源指令调整模型（如 Llama3-8B/70B, Qwen2-7B/72B），参数规模分为小型（7-8B）和中型（70-72B），使用贪婪解码（greedy decoding）确保结果可重复性，实验在 Nvidia A100 GPU 上运行。
*   **评估与分析:** 使用准确率（accuracy）、成对准确率（paired accuracy）和联合准确率（joint accuracy，仅实验 2）作为指标，并通过逻辑回归分析（logistic regression）探讨参数规模、情态条件、故事类型和提问格式对表现的影响。

## Experiment

*   **实验 1 结果（情态助动词）:** 中型参数模型（70-72B）准确率（79.6%-95.3%）显著高于小型模型（7-8B，55.1%-72.3%），且在必要性情态（necessity）上的表现优于可能性情态（possibility），表明模型处理唯一解情境的能力较强；故事类型和提问格式的影响较小且不一致，显示模型对情态语义的掌握不够稳健。
*   **实验 2 结果（态度动词）:** 中型模型准确率（60.8%-72.8%）高于小型模型（36.4%-56.9%）；高确定性动词（如 know/believe）通常表现优于低确定性动词（如 doubt），但部分中型模型（如 Llama3/Llama3.1）呈现相反趋势；事实陈述（fact-based）准确率显著高于信念陈述（belief-based），联合准确率普遍较低（最高 5.6%），表明模型在复杂情境中全面表达态度的能力有限。
*   **总体评价:** 实验设置全面，控制了多变量并通过统计分析验证结果显著性，但模型表现与人类成人（准确率 97%-100%）相比仍有差距，尤其在不确定性表达和信念推断上，生成的认知表达可能不可靠；参数规模提升带来性能改善，但非线性且不稳定。

## Further Thoughts

论文将语言学中的认知框架（如 epistemic modality 和 semantic map）引入大型语言模型评估，启发了我思考是否可以通过模仿儿童语言习得的语用环境（pragmatically informative environments）来增强模型对认知表达的学习；此外，论文提到的多模态证据和低资源语言情态表达研究方向也很有价值，是否可以利用视觉或听觉输入作为辅助证据，帮助模型理解不确定性情境？或者通过跨语言迁移学习增强非英语语言中情态语义的表征能力？