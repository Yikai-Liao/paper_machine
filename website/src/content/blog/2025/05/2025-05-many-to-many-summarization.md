---
title: "An Empirical Study of Many-to-Many Summarization with Large Language Models"
pubDatetime: 2025-05-19T11:18:54+00:00
slug: "2025-05-many-to-many-summarization"
type: "arxiv"
id: "2505.12983"
score: 0.6571135936818701
author: "grok-3-latest"
authors: ["Jiaan Wang", "Fandong Meng", "Zengkui Sun", "Yunlong Liang", "Yuxuan Cao", "Jiarong Xu", "Haoxiang Shi", "Jie Zhou"]
tags: ["LLM", "Summarization", "Multilingual", "Instruction Tuning", "Zero-Shot"]
institution: ["Pattern Recognition Center, WeChat AI, Tencent Inc, China", "Fudan University", "Waseda University", "Beijing Jiaotong University", "Zhejiang University"]
description: "本文通过重组多领域多语言数据集，系统评估了大型语言模型在多对多摘要任务上的零样本和指令微调表现，揭示其显著潜力及事实一致性挑战。"
---

> **Summary:** 本文通过重组多领域多语言数据集，系统评估了大型语言模型在多对多摘要任务上的零样本和指令微调表现，揭示其显著潜力及事实一致性挑战。 

> **Keywords:** LLM, Summarization, Multilingual, Instruction Tuning, Zero-Shot

**Authors:** Jiaan Wang, Fandong Meng, Zengkui Sun, Yunlong Liang, Yuxuan Cao, Jiarong Xu, Haoxiang Shi, Jie Zhou

**Institution(s):** Pattern Recognition Center, WeChat AI, Tencent Inc, China, Fudan University, Waseda University, Beijing Jiaotong University, Zhejiang University


## Problem Background

多对多摘要（Many-to-Many Summarization, M2MS）是一个极具挑战性的任务，旨在处理任意语言的文档并生成任意语言的摘要，结合了跨语言摘要和多语言摘要的复杂性。
随着大型语言模型（LLMs）展现出强大的多语言能力，其在 M2MS 任务上的潜力亟待探索，尤其是在多领域场景下的表现尚未被系统性研究。
关键问题在于：LLMs 是否能有效应对 M2MS 任务？零样本和指令微调两种方式的表现如何？是否存在事实一致性等潜在问题？

## Method

* **核心目标**：评估大型语言模型（LLMs）在多对多摘要（M2MS）任务上的能力，探索其在零样本和指令微调两种设置下的表现，并与传统多语言模型进行对比。
* **零样本提示（Zero-Shot Prompting）**：设计包含任务描述、领域信息和上下文示例的提示（Prompt），直接利用 LLMs 的指令跟随能力和上下文学习能力生成摘要，无需更新模型参数。这种方法测试了 LLMs 的泛化能力，提示中包含 3 个示例摘要以引导模型输出风格一致的摘要。
* **指令微调（Instruction Tuning）**：对开源 LLMs 使用重组的 M2MS 数据集（19.5K 训练样本）进行微调，以提升其任务特定能力。微调过程中采用低学习率（1e-5）和小批量大小（32），结合 DeepSpeed 优化和 Flash Attention 技术以节省计算资源，旨在验证 LLMs 在小规模任务数据上的适应性。
* **传统模型对比**：对传统多语言模型（如 mBART-50 和 PISCES）进行微调，作为基准对比。传统模型采用语言标签控制目标语言输出，输入长度限制为 1024 token，训练参数包括学习率 3e-5 和 10 个训练轮次。
* **数据处理**：从 8 个现有跨语言摘要数据集重组 47.8K 样本，覆盖 5 个领域（新闻、百科、对话、指南、技术）和 6 种语言（英语、捷克语、德语、法语、汉语、乌克兰语）。通过内在质量指标（Coverage, Redundancy, Coherence）筛选高质量样本，并控制测试集数据污染比例（小于 1%）以确保评估公平性。

## Experiment

* **零样本表现**：零样本 LLMs（如 GPT-4o）在 ROUGE-1、ROUGE-L 和 BERTScore 等指标上与微调后的传统模型（如 PISCES）表现相当，显示出强大的指令跟随和上下文学习能力。例如，GPT-4o 的 ROUGE-1 得分为 26.0，接近 PISCES 的 30.8。
* **指令微调效果**：指令微调显著提升了开源 LLMs 的 M2MS 能力，Vicuna-13B-16k 在微调后 ROUGE-1 达到 38.0，远超 PISCES 的 30.8 以及零样本 GPT-4o 的 26.0，表明 LLMs 在小规模任务数据上的适应性优于传统模型，且在所有领域和语言对上均有提升。
* **通用能力验证**：通过 MMLU 数据集测试，指令微调未牺牲 LLMs 的通用任务解决能力，部分模型（如 LLaMa-2-13B）甚至略有提升。
* **实验设置**：实验覆盖 18 个 LLMs 和 2 个传统模型，数据涉及多领域和多语言，评估指标包括 ROUGE、BERTScore 和 GPT-4o 评分（Conciseness, Coherence, Relevance），结合自动和人工评估，设置较为全面合理。但测试样本仅随机抽取 500 个用于 GPT-4o 评分，可能存在一定偏差。
* **局限性**：人工评估显示 LLMs 存在事实一致性问题，指令微调后幻觉（Hallucination）和细节错误（Particulars Error）比例增加，例如微调后 LLaMa-2-13B 的幻觉错误比例从 17% 升至 23%，需进一步关注。

## Further Thoughts

论文揭示了指令微调在小规模任务数据上显著提升 LLMs 任务特定能力的潜力，同时不牺牲通用能力，这启发我们思考是否可以通过更精细的指令设计或数据选择策略减少幻觉问题，例如在微调数据中引入事实一致性约束或结合外部知识库进行验证。此外，LLMs 对长文档的支持能力（可处理数千 token）相比传统模型的限制（1K token）提示我们可以在长文档摘要任务中探索更多实际应用场景，如多语言法律文档或学术论文摘要。