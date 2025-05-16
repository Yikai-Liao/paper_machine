---
title: "Variational Prefix Tuning for Diverse and Accurate Code Summarization Using Pre-trained Language Models"
pubDatetime: 2025-05-14T01:46:56+00:00
slug: "2025-05-variational-prefix-tuning"
type: "arxiv"
id: "2505.09062"
score: 0.5882105325952485
author: "grok-3-latest"
authors: ["Junda Zhao", "Yuliang Song", "Eldan Cohen"]
tags: ["LLM", "Code Summarization", "Diversity Generation", "Parameter Efficiency", "Pre-Training"]
institution: ["University of Toronto"]
description: "本文提出变分前缀调优（VPT），通过结合 CVAE 和前缀调优，参数高效地增强预训练模型生成多样且准确的代码摘要的能力，并在多个数据集和模型上显著优于基线方法。"
---

> **Summary:** 本文提出变分前缀调优（VPT），通过结合 CVAE 和前缀调优，参数高效地增强预训练模型生成多样且准确的代码摘要的能力，并在多个数据集和模型上显著优于基线方法。 

> **Keywords:** LLM, Code Summarization, Diversity Generation, Parameter Efficiency, Pre-Training

**Authors:** Junda Zhao, Yuliang Song, Eldan Cohen

**Institution(s):** University of Toronto


## Problem Background

代码摘要生成（Code Summarization）是软件工程中的重要任务，旨在将复杂源代码转化为简洁的人类可读描述，以提升代码可读性和维护性。
当前基于大型语言模型（LLMs）或代码专用模型（LLMCs）的方法通常只生成单一摘要，忽略了多样性（Diversity）的重要性，当生成的摘要不准确或不合适时，用户缺乏替代选择。
现有多样性生成方法（如束搜索或采样）要么差异过小，要么牺牲准确性，因此亟需一种能在保持准确性前提下生成多样化摘要的方法。

## Method

*   **核心思想:** 提出变分前缀调优（Variational Prefix Tuning, VPT），通过结合条件变分自编码器（Conditional Variational AutoEncoder, CVAE）和前缀调优（Prefix Tuning），增强预训练模型生成多样且准确代码摘要的能力。
*   **具体实现:** 
    *   将 CVAE 框架作为模块化组件集成到预训练模型中，通过学习目标摘要的分布，生成连续的变分前缀（Variational Prefixes），这些前缀在解码时引导模型生成多样化输出。
    *   使用预训练模型（如 CodeT5+）的冻结编码器生成源代码的上下文嵌入，作为先验分布（Prior Distribution）的均值，提升生成的前缀与代码语义的相关性。
    *   在训练阶段，通过优化证据下界（ELBO），平衡重建损失（确保准确性）和 KL 散度（确保多样性）；在推理阶段，从先验分布采样前缀，结合束搜索（Beam Search）提升单个摘要质量。
    *   引入双准则子集选择（Bi-Criteria Subset Selection），从大量候选摘要中筛选出兼具多样性和准确性的子集，通过优化质量（基于模型预测概率）和多样性（基于 BLEU 距离）目标实现。
*   **参数效率:** VPT 仅训练少量参数（约占完整微调的 10.8%），避免了对大型模型的昂贵重新训练，体现了参数高效调优（PEFT）的优势。
*   **关键创新:** 通过变分方法建模目标分布实现多样性，同时利用前缀机制保持预训练模型的原始性能，并通过双准则优化最终输出。

## Experiment

*   **有效性:** VPT 在 Java 和 Python 数据集上显著优于基线方法（如束搜索、采样、随机束搜索和多样性束搜索），在 Oracle 准确性指标（如 BLEU, ROUGE-L, METEOR, BERTScore）上提升明显，例如在 Python 数据集上 BLEU 从束搜索的 44.28 提升至 46.40（#U=10），且随摘要数量增加（#U=20）提升更显著（从 45.63 到 48.62）。
*   **多样性:** 在多样性指标（如 Distinct-1, Distinct-2 和 Self-BLEU）上，VPT 接近采样方法的多样性，但准确性更高，展现了多样性与质量的更好平衡。
*   **适应性:** VPT 成功应用于多种预训练模型（如 CodeT5+, PLBART, NeuralCodeSum），并优于 LoRA 微调的 CodeLlama 和 GPT-4o（结合少样本学习），证明了其跨模型的适应性。
*   **实验设置合理性:** 实验涵盖了两个主流数据集（Java 和 Python）、多种评估指标（准确性和多样性）、多个基线对比以及消融研究（验证各组件贡献），设置全面且严谨。
*   **计算开销:** VPT 训练参数少，推理时仅增加少量前缀 token 的计算负担，效率较高。

## Further Thoughts

VPT 通过变分方法（如 CVAE）建模目标分布来实现多样性生成的思路，不仅适用于代码摘要，还可能推广至其他需要多样输出的生成任务（如代码生成、对话系统），启发我们探索更多基于分布建模的多样性策略。
此外，VPT 的参数高效特性提示我们可以尝试将其与其他 PEFT 方法（如 LoRA 或 Adapter）结合，进一步优化计算成本或性能。
双准则子集选择机制通过平衡质量和多样性筛选输出，这种多目标优化策略可能对其他生成任务或决策问题有借鉴意义，未来或许可以引入强化学习动态调整选择策略。