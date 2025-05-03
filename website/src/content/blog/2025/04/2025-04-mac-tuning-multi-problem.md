---
title: "MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness"
pubDatetime: 2025-04-30T16:17:53+00:00
slug: "2025-04-mac-tuning-multi-problem"
type: "arxiv"
id: "2504.21773"
score: 0.6613055413730277
author: "grok-3-latest"
authors: ["Junsheng Huang", "Zhitao He", "Sandeep Polisetty", "Qingyun Wang", "May Fung"]
tags: ["LLM", "Confidence Estimation", "Hallucination Mitigation", "Multi-Task Reasoning"]
institution: ["Hong Kong University of Science and Technology", "University of Illinois", "UMass Amherst"]
description: "本文提出MAC-Tuning方法，通过分离答案预测和置信度估计的学习过程，显著提升大型语言模型在多问题设置下的知识边界意识和推理可靠性，平均精度最高提升25%。"
---

> **Summary:** 本文提出MAC-Tuning方法，通过分离答案预测和置信度估计的学习过程，显著提升大型语言模型在多问题设置下的知识边界意识和推理可靠性，平均精度最高提升25%。 

> **Keywords:** LLM, Confidence Estimation, Hallucination Mitigation, Multi-Task Reasoning

**Authors:** Junsheng Huang, Zhitao He, Sandeep Polisetty, Qingyun Wang, May Fung

**Institution(s):** Hong Kong University of Science and Technology, University of Illinois, UMass Amherst


## Problem Background

大型语言模型（LLMs）在知识密集型任务中常生成不存在的事实（即幻觉问题），尤其是在回答超出其参数化知识边界的问题时。
现有研究主要聚焦于单问题设置（逐一回答独立问题），而对多问题设置（单一输入中处理多个子问题）的研究不足。
多问题设置因涉及复杂的上下文区分和推理综合，容易导致上下文混淆和推理错误传播，进而加剧幻觉问题，亟需提升模型的知识边界意识和置信度估计能力。

## Method

*   **核心思想**：提出Multiple Answers and Confidence Stepwise Tuning（MAC-Tuning）方法，通过分离答案预测和置信度估计的学习过程，增强LLMs在多问题设置下的知识边界意识，减少幻觉并提高回答可靠性。
*   **数据构建**：从原始数据集中随机组合多个单问题，构建多问题数据集；通过比较模型输出与真实答案，识别知识边界并标注置信度（‘I am sure’或‘I am unsure’）；基于此，构建两种训练数据对：多问题-答案对（Multiple QA pair）和多问题-答案-置信度对（Multiple QA-Confidence pair）。
*   **两步训练**：采用两步监督微调策略；第一步基于多问题-答案对训练模型生成正确答案，优化答案预测能力；第二步基于多问题-答案-置信度对训练模型表达置信度，优化置信度估计能力；这种分离学习避免了答案生成和置信度估计之间的干扰。
*   **推理过程**：在推理时，模型同时输出答案和对每个答案的置信度评估，帮助用户判断回答的可信度。
*   **关键创新**：针对多问题设置的复杂性，强调知识边界意识，并通过分步学习策略提升模型在多任务推理中的表现。

## Experiment

*   **有效性**：MAC-Tuning在独立问题设置（CoQA, ParaRel, GSM, MMLU）和序列问题设置（MTI-Bench, SQA）数据集上均取得最佳平均精度（AP）分数，最高提升达25%，期望校准误差（ECE）显著降低，表明模型能更准确区分确定和不确定问题；准确率平均提升23.7%，最高达45.8%。
*   **优越性**：相较于基线方法（如QA-Only, Single-QA, Merge-AC），MAC-Tuning在置信度估计和答案准确性上均表现更优，尤其是在分离学习答案和置信度的策略下，相较于不分离的Merge-AC，AP提升达11%-25%。
*   **全面性与合理性**：实验在不同基模型（如LLaMA3-8B-Instruct, Qwen2-7B-Instruct）上验证了方法的鲁棒性；跨域测试和不同问题数量分析显示方法具有较好的泛化能力；但在较难数据集（如MMLU）上，随着问题数量增加，性能略有下降，反映出模型处理多重复杂任务的局限。
*   **开销**：主要增加了数据构建和两步微调的计算成本，但通过使用LoRA等参数高效微调方法，整体开销可控。

## Further Thoughts

MAC-Tuning通过置信度估计增强知识边界意识的思路可扩展至检索增强生成（RAG），结合外部知识库动态调整置信度以进一步减少幻觉；分离学习答案和置信度的策略启发多任务学习中更细粒度的任务分解，避免任务干扰；此外，模型对不同问题数量表现的差异提示未来可设计自适应机制，根据任务难度和数量动态调整推理策略。