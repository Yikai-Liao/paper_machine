---
title: "Variational Prefix Tuning for Diverse and Accurate Code Summarization Using Pre-trained Language Models"
pubDatetime: 2025-05-14T01:46:56+00:00
slug: "2025-05-variational-prefix-tuning"
type: "arxiv"
id: "2505.09062"
score: 0.5882105325952485
author: "grok-3-latest"
authors: ["Junda Zhao", "Yuliang Song", "Eldan Cohen"]
tags: ["LLM", "Code Summarization", "Diversity", "Sampling", "Pre-Training"]
institution: ["University of Toronto"]
description: "本文提出变分前缀调优（VPT）方法，通过参数高效地集成条件变分自编码器到预训练代码摘要模型中，显著提升了生成摘要的多样性和准确性。"
---

> **Summary:** 本文提出变分前缀调优（VPT）方法，通过参数高效地集成条件变分自编码器到预训练代码摘要模型中，显著提升了生成摘要的多样性和准确性。 

> **Keywords:** LLM, Code Summarization, Diversity, Sampling, Pre-Training

**Authors:** Junda Zhao, Yuliang Song, Eldan Cohen

**Institution(s):** University of Toronto


## Problem Background

代码摘要生成是软件工程中的关键任务，旨在将复杂源代码转化为简洁的人类可读描述，以提升代码可读性和维护性。
现有基于大型语言模型（LLMs）或代码专用模型（LLMCs）的方法通常只生成单一摘要，忽略了多样性（Diversity）的重要性：如果生成的摘要不准确或不合适，开发者别无选择。
此外，现有多样性生成方法（如随机采样）往往牺牲准确性，而传统方法（如束搜索）生成的多个摘要差异较小。
因此，论文试图解决如何在不牺牲准确性的前提下，生成一组多样且准确的代码摘要，增加至少一个摘要合适的概率。

## Method

*   **核心思想:** 提出变分前缀调优（Variational Prefix Tuning, VPT），通过将条件变分自编码器（Conditional Variational Autoencoder, CVAE）以参数高效的方式集成到预训练代码摘要模型中，增强模型生成多样且准确摘要的能力。
*   **实现细节:** 
    *   借鉴前缀调优（Prefix Tuning）的思想，VPT 学习一组连续的前缀向量来引导生成过程，而无需重新训练整个模型。
    *   使用 CVAE 建模目标摘要的分布，通过条件先验分布（Prior Distribution）和后验分布（Posterior Distribution）生成随机的连续前缀，在解码时引导模型输出多样化摘要。
    *   利用预训练模型（如 CodeT5+）的编码器生成上下文嵌入，参数化先验和后验分布；在训练时优化证据下界（ELBO），平衡重建损失和分布正则化；在推理时从先验分布采样潜在变量作为前缀。
    *   引入束搜索（Beam Search）提升每个前缀生成的摘要质量，并通过双准则子集选择（Bi-Criteria Subset Selection）从大量候选摘要中优化挑选，平衡质量和多样性。
*   **优势:** 参数高效，仅需训练少量参数（约占全微调的10.8%），适合资源受限场景；同时兼顾多样性和准确性，避免了传统方法的局限。

## Experiment

*   **有效性:** VPT 在 Java 和 Python 数据集上，基于多个预训练模型（如 CodeT5+、PLBART）测试，显著优于基线方法（如 Beam Search、Sampling、Stochastic Beam Search、Diverse Beam Search），尤其在生成更多摘要（如 20 个）时提升更明显，例如在 CodeT5+ 上，Python 数据集的 Oracle BLEU 分数从 Beam Search 的 45.63 提升至 48.62。
*   **多样性:** 在多样性指标（如 Distinct N-gram、Self-BLEU）上，VPT 仅次于 Sampling，但结合准确性指标，VPT 实现了更好的平衡，而 Sampling 准确性较低。
*   **适应性:** VPT 成功应用于多种预训练模型，证明了通用性；与 LoRA 微调的 CodeLlama 和 GPT-4o 相比，VPT 在大多数指标上表现更优。
*   **实验设置合理性:** 实验覆盖多个维度（不同模型、数据集、生成数量），评价指标包括准确性（BLEU、ROUGE、METEOR、BERTScore）和多样性（Distinct N-gram、Self-BLEU），较为全面；消融研究验证了各组件贡献，统计检验（Wilcoxon 检验）确认结果显著性。
*   **局限性:** 数据集仅限于 Java 和 Python，未覆盖其他语言；评价指标可能无法完全反映语义质量，缺乏足够人工评估。

## Further Thoughts

VPT 将生成模型（如 CVAE）与预训练模型结合，通过潜在空间采样实现多样性输出的思路，可推广至其他代码任务（如代码生成）或自然语言处理任务（如对话系统），解决单一输出不足的问题；
参数高效微调（PEFT）与生成模型结合的策略启发我们探索更多方法（如 LoRA、Adapter）以降低计算成本；
双准则子集选择方法提示可以在生成后处理阶段通过多目标优化提升结果实用性；
100 个候选摘要的 Oracle 分数远高于最终结果，表明更好的子集选择或重排序算法（如引入强化学习或用户反馈）有巨大潜力。