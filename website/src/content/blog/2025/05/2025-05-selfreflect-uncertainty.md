---
title: "Self-reflective Uncertainties: Do LLMs Know Their Internal Answer Distribution?"
pubDatetime: 2025-05-26T17:59:53+00:00
slug: "2025-05-selfreflect-uncertainty"
type: "arxiv"
id: "2505.20295"
score: 0.7021710220673909
author: "grok-3-latest"
authors: ["Michael Kirchhof", "Luca Füger", "Adam Goliński", "Eeshan Gunesh Dhekane", "Arno Blaas", "Sinead Williamson"]
tags: ["LLM", "Uncertainty Quantification", "Sampling", "Summarization", "Predictive Sufficiency"]
institution: ["Apple", "Independent Researcher"]
description: "本文提出SelfReflect度量方法，通过掩码任务评估总结字符串是否忠实反映大型语言模型的内部回答分布，为不确定性表达开辟新路径，并揭示当前模型在自省不确定性上的局限。"
---

> **Summary:** 本文提出SelfReflect度量方法，通过掩码任务评估总结字符串是否忠实反映大型语言模型的内部回答分布，为不确定性表达开辟新路径，并揭示当前模型在自省不确定性上的局限。 

> **Keywords:** LLM, Uncertainty Quantification, Sampling, Summarization, Predictive Sufficiency

**Authors:** Michael Kirchhof, Luca Füger, Adam Goliński, Eeshan Gunesh Dhekane, Arno Blaas, Sinead Williamson

**Institution(s):** Apple, Independent Researcher


## Problem Background

大型语言模型（LLMs）在面对不确定性时（如问题模糊或事实不确定），通常只能通过数值或简单语言化描述来表达置信度，无法全面揭示其内部对可能回答的分布情况。
这种局限性限制了用户对模型信念的深入理解，例如无法得知模型认为哪些其他答案是可能的，以及支持这些答案的相关信息。

## Method

*   **核心思想:** 提出一种名为SelfReflect的度量方法，用于评估一个总结字符串是否忠实地反映了LLM内部对回答的概率分布，基于信息论中的预测充分性（predictive sufficiency）概念。
*   **具体实现:** 
    *   使用一个辅助的评判模型（judge LLM）执行掩码任务（masked-token prediction task），即预测被掩码的单词。
    *   分别基于总结字符串和从LLM内部分布中采样的多个回答，生成两种预测分布。
    *   使用1-Wasserstein距离量化两种预测分布之间的差异，作为总结字符串质量的度量。
    *   为提高效率，排除停用词并使用温度调整（temperature scaling）来平滑分布，考虑同义词影响。
*   **总结生成策略:** 
    *   测试了‘采样并总结’（Sample & Summarize）方法，即先从模型中采样多个回答，然后生成总结。
    *   测试了‘单次解码’（Single-decoding）方法，包括贪婪解码（Greedy）、基础提示（Basic）和链式推理（Chain-of-Thought, CoT），试图直接生成反映分布的总结。
*   **关键点:** SelfReflect不依赖于特定模型或任务，旨在成为通用的不确定性表达评估工具，同时探索LLM是否能自省并表达其内部不确定性。

## Experiment

*   **有效性:** SelfReflect在多个数据集（如Natural Questions, TriviaQA, MMLU）上表现出色，能在99.8%的情况下区分好的总结和差的总结，在94.2%的情况下区分好的和‘几乎好的’总结，尤其在细粒度区分（如详细程度和相对概率）上优于基线方法（如LM Judge、Embedding距离）。
*   **一致性:** 与人类判断高度一致，Krippendorff’s α值为0.690，接近人类间一致性（0.723），表明其评估结果具有实际意义。
*   **总结生成效果:** ‘采样并总结’方法显著优于单次解码方法，SelfReflect分数更低（即更忠实），表明当前LLM难以通过单次解码自发表达内部分布；即使是强化学习训练的推理模型（RLVR）也未展现明显改进。
*   **实验设置合理性:** 实验覆盖了多种模型（Qwen2.5系列、Llama系列、Phi 4等）和数据集，测试了不同规模模型的表现，确保结果的广泛适用性；评判模型的选择（如Qwen2.5 7B）通过对比实验证明鲁棒性。
*   **额外发现:** ‘采样并总结’生成的总结在覆盖真实答案方面优于贪婪解码（RougeL-Recall分数65.6% vs 59.5%），表明分布信息可能比单一输出更接近真相。

## Further Thoughts

SelfReflect作为一个通用的不确定性评估工具，启发我们可以在模型训练中引入分布表达目标，例如通过强化学习优化总结生成能力，或在交互式应用中利用总结字符串帮助用户理解模型的不确定性；此外，论文揭示了分布信息比单一输出更接近真实答案的潜力，提示未来可以探索多模态分布的采样和表达方法，以提升模型的可信度和实用性。