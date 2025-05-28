---
title: "The Price of Format: Diversity Collapse in LLMs"
pubDatetime: 2025-05-25T02:52:35+00:00
slug: "2025-05-diversity-collapse-llms"
type: "arxiv"
id: "2505.18949"
score: 0.7857524807941652
author: "grok-3-latest"
authors: ["Longfei Yun", "Chenyang An", "Zilong Wang", "Letian Peng", "Jingbo Shang"]
tags: ["LLM", "Instruction Tuning", "Diversity Collapse", "Prompt Design", "Output Variability"]
institution: ["University of California, San Diego"]
description: "本文揭示了指令微调中结构化提示模板导致的多样性崩溃现象，并通过消融实验和缓解策略证明移除结构化元素能有效提升输出多样性，为提示设计和模型训练提供了重要指导。"
---

> **Summary:** 本文揭示了指令微调中结构化提示模板导致的多样性崩溃现象，并通过消融实验和缓解策略证明移除结构化元素能有效提升输出多样性，为提示设计和模型训练提供了重要指导。 

> **Keywords:** LLM, Instruction Tuning, Diversity Collapse, Prompt Design, Output Variability

**Authors:** Longfei Yun, Chenyang An, Zilong Wang, Letian Peng, Jingbo Shang

**Institution(s):** University of California, San Diego


## Problem Background

指令微调的大型语言模型（LLMs）在推理时常使用结构化提示模板（如角色标记和特殊标记）以确保输出一致性和对齐性，但这些模板导致了‘多样性崩溃’（Diversity Collapse）现象，即模型在开放式生成任务中输出语义和主题上的多样性显著下降，限制了创意性和表达变异性。
这一问题源于指令微调过程中模型对结构化格式的强生成先验内化，可能进一步受到训练数据分布偏差和目标函数设计的影响。

## Method

*   **核心思想：** 探究结构化提示模板对输出多样性的影响，并通过调整提示格式和训练策略寻找缓解多样性崩溃的方法。
*   **提示格式消融实验：** 设计了四种提示模式，从高度结构化的‘Full Template’（包含系统、用户、助手标记等特殊标记和对话格式）到完全无结构的‘Simple Steer’（仅包含任务指令），逐步移除结构化元素（如角色标记、系统消息），以隔离各元素对多样性的影响。
*   **模型与任务选择：** 在多个指令微调模型（如Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct等）上测试，任务涵盖常识推理（CommonGen, ELI5）、故事补全（WritingPrompts, ROCStory）和开放式生成（新闻、旅行推荐等），确保结果的普适性和鲁棒性。
*   **多样性度量：** 采用语义多样性（基于句子嵌入的平均成对距离）和标签多样性（基于生成主题的熵）作为主要指标，辅以传统指标（如Distinct-N和self-BLEU）验证结果，全面评估输出变异性。
*   **缓解策略：** 尝试多种方法，包括混合模板（Mixed Template，随机选择不同格式）、自然指令（Natural Instruction，训练和推理均无结构化格式）、混合训练（Mixed Training，加入无结构预训练数据）、调整解码温度（Temperature Scaling）以及显式提示多样性（Explicit Prompting for Diversity），以探索恢复多样性的可行性。
*   **下游任务分析：** 评估不同提示格式对结构敏感任务（如GSM8K, IFEval）和知识密集型任务（如MMLU, WebQuestions）性能的影响，分析一致性和多样性之间的权衡。

## Experiment

*   **多样性崩溃验证：** 实验结果（Table 1, Figure 2）表明，结构化模板（Full Template）在所有模型和任务中显著降低输出多样性，而无结构提示（Simple Steer）提升了语义和主题多样性，例如Llama-3-8B-Instruct在新闻生成任务中的主题熵从0.0538提升至0.1399。
*   **消融分析效果：** 逐步移除结构化元素（Table 2, Figure 5）后发现，即使是轻量级结构（如Minimum Dialog）也会限制多样性，只有完全无结构的Simple Steer能最大程度恢复多样性，表明结构化模板导致的‘行为锚定’（Behavioral Anchoring）是主要原因。
*   **缓解策略表现：** Natural Instruction和Simple Steer在多样性上表现最佳（Table 3），而Mixed Template和Mixed Training改进有限；显式提示多样性（Table 5）虽有提升，但仍不及Simple Steer；高温度解码（Figure 7）增加多样性，但受模板限制效果较小。
*   **下游任务影响：** 结构化模板对结构敏感任务（如GSM8K, IFEval）有益，但对知识密集型任务（如MMLU）可能有害（Table 4）；训练和推理格式一致性对性能影响更大。
*   **实验设置评价：** 实验覆盖多种模型、任务和指标，设置全面合理，但未探讨其他提示策略（如思维链提示）的影响，且多样性度量局限于语义和词汇层面，未深入话语层面。

## Further Thoughts

本文揭示了结构化提示在一致性和多样性之间的权衡，启发我们设计‘动态提示’策略，根据任务类型自适应调整格式；同时，交叉熵损失导致多样性下降的问题提示我们探索熵正则化或多样性奖励的训练目标；此外，结构化模板的‘行为锚定’效应让我思考是否可以开发‘多样性导向解码’方法，在生成早期动态调整概率分布，避免过早收敛；最后，指令微调引入的隐性偏见（如风格一致性）也值得进一步研究其对模型长期行为的影响。