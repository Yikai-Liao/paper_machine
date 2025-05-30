---
title: "CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models"
pubDatetime: 2025-05-27T06:16:27+00:00
slug: "2025-05-cognibench-faithfulness"
type: "arxiv"
id: "2505.20767"
score: 0.49218164339463916
author: "grok-3-latest"
authors: ["Xiaqiang Tang", "Jian Li", "Keyu Hu", "Du Nan", "Xiaolong Li", "Xi Zhang", "Weigao Sun", "Sihong Xie"]
tags: ["LLM", "Faithfulness Hallucination", "Cognitive Statements", "Automated Annotation", "Reasoning"]
institution: ["The Hong Kong University of Science and Technology (Guangzhou)", "Hunyuan AI Digital Human, Tencent", "Beijing University of Posts and Telecommunications", "Shanghai AI Laboratory"]
description: ""
---

> **Summary:**  

> **Keywords:** LLM, Faithfulness Hallucination, Cognitive Statements, Automated Annotation, Reasoning

**Authors:** Xiaqiang Tang, Jian Li, Keyu Hu, Du Nan, Xiaolong Li, Xi Zhang, Weigao Sun, Sihong Xie

**Institution(s):** The Hong Kong University of Science and Technology (Guangzhou), Hunyuan AI Digital Human, Tencent, Beijing University of Posts and Telecommunications, Shanghai AI Laboratory


## Problem Background

大型语言模型（LLMs）在生成内容时常出现‘忠实性幻觉’（Faithfulness Hallucination），即生成内容与提供上下文不一致的问题。现有基准数据集主要关注事实性陈述（Factual Statements），忽略了需要推理、评价和解释的认知性陈述（Cognitive Statements），导致缺乏针对认知性陈述忠实性的标准化评估框架和数据支持，尤其在医疗、法律等高风险领域应用中至关重要。此外，手动标注成本高昂，无法适应快速更新的LLM模型需求。

## Method

* **CogniBench数据集与法律启发框架**：构建了一个多轮知识基础对话数据集CogniBench，通过手动标注提供句级别的忠实性标签，特别针对认知性陈述。受法律证据评估启发，提出三个递增严格的忠实性标准：Rational（合理推测，允许合理但无法验证的推测）、Grounded（上下文支持，确保逻辑上由上下文推导）和Unequivocal（无争议结论，无其他合理解释），以减少评估主观性并适应不同应用场景需求。
* **自动化标注扩展（CogniBench-L）**：为解决手动标注成本问题，设计自动化标注流程，利用GPT-4作为标注工具，通过对比提示（Contrastive Prompting，提供正反例减少歧义）和形成性提示（Formative Prompting，一次性标注整段对话中的每个句子）提高标注准确性，并采用多响应采样（Multi-response Sampling，多次提示后取多数投票结果）增强可靠性，最终生成大规模数据集CogniBench-L（超过24k个对话）。
* **CogniDet模型开发**：基于CogniBench-L微调一个8B参数模型CogniDet，用于检测事实性和认知性陈述中的幻觉，采用单次前向计算以降低成本，输入为上下文-对话对，直接输出幻觉句子列表。

## Experiment

* **认知性陈述动态与幻觉率**：实验显示，随着对话轮数增加，认知性陈述比例从15%升至50%，其幻觉率高达64.8%，远高于事实性陈述的13.9%。不同模型表现差异显著，如GPT-4认知性陈述幻觉率为60.1%，Gemini-Pro高达79.9%。
* **自动化标注效果**：自动化标注流程在CogniBench上的F1分数达82.2%，接近人类标注水平，证明其作为评估新模型的可靠代理。
* **CogniDet性能提升**：CogniDet整体F1分数为70.3%，在认知性陈述检测上达73.8%，显著优于现有方法（如RAGTruth的11.2%和FAVA的5.1%），且计算成本更低（单次前向计算对比NLI基线的多次计算）。
* **实验设置合理性**：实验涵盖多轮对话、多种模型和检测方法对比，设置全面，但数据集主要基于常识性开放领域对话，缺乏对高风险领域（如医疗、金融）的深入覆盖，可能限制泛化性。

## Further Thoughts

法律证据评估逻辑的引入为减少认知性陈述评估主观性提供了系统性思路，可进一步扩展到伦理决策或政策分析等需要严谨推理的场景；多响应采样策略在减少LLM标注幻觉方面的成功应用，启发其在其他高质量标注任务（如情感分析）中的潜力；对话轮数增加导致认知性陈述比例和幻觉率上升的动态，提示在长上下文对话系统中可通过动态提示调整或上下文约束缓解后期幻觉问题。