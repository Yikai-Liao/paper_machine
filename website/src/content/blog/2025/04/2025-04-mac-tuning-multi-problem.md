---
title: "MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness"
pubDatetime: 2025-05-01T15:49:13Z
slug: "2025-04-mac-tuning-multi-problem"
type: "arxiv"
id: "2504.21773"
score: 0.8404728375184152
author: "grok-3-latest"
authors: ["Junsheng Huang", "Zhitao He", "Sandeep Polisetty", "Qingyun Wang", "May Fung"]
tags: ["LLM", "Confidence Calibration", "Multi-Problem Setting", "Fine-Tuning", "Hallucination Mitigation"]
institution: ["Hong Kong University of Science and Technology", "University of Illinois", "UMass Amherst"]
description: "本文提出 MAC-Tuning 方法，通过分离答案预测和置信度估计的学习过程，显著提升了大型语言模型在多问题设置下的推理可靠性和幻觉抑制能力。"
---

> **Summary:** 本文提出 MAC-Tuning 方法，通过分离答案预测和置信度估计的学习过程，显著提升了大型语言模型在多问题设置下的推理可靠性和幻觉抑制能力。 

> **Keywords:** LLM, Confidence Calibration, Multi-Problem Setting, Fine-Tuning, Hallucination Mitigation
> **Recommendation Score:** 0.8404728375184152

**Authors:** Junsheng Huang, Zhitao He, Sandeep Polisetty, Qingyun Wang, May Fung
**Institution(s):** Hong Kong University of Science and Technology, University of Illinois, UMass Amherst

## Problem Background

大型语言模型（LLMs）在知识密集型任务中常生成不存在的事实（即幻觉），尤其是在回答超出其参数化知识边界的问题时。
现有研究主要聚焦于单问题设置（single-problem setting），而对于更具挑战性的多问题设置（multi-problem setting）——即一次输入包含多个子问题需同时准确回答——的研究较少。
多问题设置因上下文干扰和推理复杂性，显著增加了模型幻觉风险，而这种设置在实际应用中日益普遍（如共享上下文任务、降低API成本），亟需提升模型置信度估计和推理可靠性。

## Method

*   **核心思想**：提出 Multiple Answers and Confidence Stepwise Tuning（MAC-Tuning）方法，通过分离答案预测和置信度估计的学习过程，增强模型对知识边界的感知能力，减少多问题设置下的幻觉。
*   **数据构建**：从原始数据集中随机组合多个单问题，形成多问题数据集；通过比较模型输出与真实答案，识别知识边界，将问题标注为‘确定’（I am sure）或‘不确定’（I am unsure）；构建两种训练数据：多问题问答对（Multiple QA pair）和多问题问答-置信度对（Multiple QA-Confidence pair）。
*   **两步训练**：采用两阶段监督微调，第一步优化模型生成正确答案，使用目标函数 max log P(A|Q; Θ0)；第二步优化模型表达置信度，使用目标函数 max log P(C|Q, A; Θ1）。这种分离训练避免了模型同时学习答案和置信度时的混淆，提升了多问题设置下的性能。
*   **推理过程**：在推理时，模型基于训练的置信度表达，区分可靠和不可靠的回答，从而减少幻觉。
*   **关键创新**：针对多问题设置的复杂性设计，将置信度估计与答案生成解耦，使模型更清晰地认识自己的知识边界，同时保持对多问题综合推理的能力。

## Experiment

*   **有效性**：MAC-Tuning 在独立问题设置（CoQA, ParaRel, GSM, MMLU）和顺序问题设置（MTI-Bench, SQA）数据集上均取得最佳平均精度（AP），最高提升达 25%，期望校准误差（ECE）显著降低，表明模型置信度校准更优；准确率平均提升 23.7%，最高达 45.8%。
*   **优越性**：相比基线方法（如 QA-Only, Single-QA, Merge-AC），MAC-Tuning 在置信度校准和准确率上均有显著优势，尤其在分离答案和置信度学习的设计上表现出关键作用。
*   **实验全面性**：实验覆盖多种数据集和问题设置，并在不同基模型（如 LLaMA3-8B 和 Qwen2-7B）上验证了方法的鲁棒性；跨域测试和不同问题数量分析进一步证明了方法的泛化能力。
*   **局限性**：由于成本和资源限制，实验范围有限，可能未完全揭示方法在更广泛数据集或模型上的表现；提示词的微小变化可能影响结果，提示实际应用中需关注提示工程的稳定性。

## Further Thoughts

MAC-Tuning 的分离训练思路启发我们思考：是否可以将答案生成和置信度估计的解耦扩展到其他生成式任务（如事实性验证），并通过更复杂的置信度表达形式（如概率分布）进一步提升效果？
多问题设置在教育、客服等领域有广泛应用前景，是否可以结合检索增强生成（RAG）技术减少上下文干扰？
此外，是否可以设计动态机制，让模型在推理时实时调整对知识边界的感知，适应不同用户或任务需求？