---
title: "Bye-bye, Bluebook? Automating Legal Procedure with Large Language Models"
pubDatetime: 2025-05-05T16:18:07+00:00
slug: "2025-05-bluebook-automation"
type: "arxiv"
id: "2505.02763"
score: 0.5863059716538778
author: "grok-3-latest"
authors: ["Matthew Dahl"]
tags: ["LLM", "Legal Automation", "Procedural Rules", "Citation Formatting", "Context Learning"]
institution: ["Yale Law School"]
description: "本文通过构建 Bluebook 任务数据集并测试大型语言模型的程序遵循能力，揭示了其在法律引用格式自动化中的局限性，为法律程序AI研究提供了重要基准。"
---

> **Summary:** 本文通过构建 Bluebook 任务数据集并测试大型语言模型的程序遵循能力，揭示了其在法律引用格式自动化中的局限性，为法律程序AI研究提供了重要基准。 

> **Keywords:** LLM, Legal Automation, Procedural Rules, Citation Formatting, Context Learning

**Authors:** Matthew Dahl

**Institution(s):** Yale Law School


## Problem Background

法律实践要求严格遵守程序规则，而美国法律引用格式《Bluebook: A Uniform System of Citation》以其复杂性（超过500页规则）成为法律从业者的负担。
论文探讨大型语言模型（LLMs）是否能自动化这一程序性任务，减轻人工负担，并为更广泛的法律程序自动化提供初步证据，解决当前对 LLMs 在法律程序领域能力评估不足的问题。

## Method

*   **数据集构建**：作者创建了一个包含866个 Bluebook 格式化任务的原创数据集，任务分为案例法、成文法和其他法律资源引用三类，形式包括完形填空（Cloze）和开放式生成（Open-Ended），答案基于法律写作教材确保权威性。
*   **模型测试**：选取五款旗舰 LLMs（OpenAI 的 GPT 4.1、Anthropic 的 Claude 3.5 Sonnet、Google 的 Gemini 2.5 Flash、Meta 的 Llama 3.1 405B、DeepSeek 的 V3 0324），在零样本（Zero-Shot）设置下评估其生成符合 Bluebook 规则引用的能力，测试模型是否能不经额外训练直接应用规则。
*   **上下文学习实验**：针对 Gemini 2.5 Flash，额外提供 Bluebook 规则（通过公共领域 Indigo Book，约90k tokens）进行上下文学习（In-Context Learning），观察规则输入是否显著提升准确率，测试长上下文处理能力。
*   **评估方式**：采用精确字符串匹配判断引用是否完全符合 Bluebook 规则，对次要格式错误（如斜体）宽松处理，并通过 Levenshtein 编辑距离分析错误严重性，同时对记忆驱动性能进行补充测试（如合成案例）。
*   **设计目标**：方法旨在隔离 LLMs 的程序规则遵循能力，避免其依赖训练数据中的记忆或实质性法律推理，聚焦于格式化任务的准确性。

## Experiment

*   **零样本性能**：五款 LLMs 在 Bluebook 任务上的平均准确率介于69%-74%，GPT 4.1 表现最佳（74%），Llama 3.1 405B 最低（69%）；在案例法任务上表现较好（平均83%），但在成文法（41%）和其他任务（34%）上准确率显著下降，表明模型对复杂规则的适应性不足。
*   **上下文学习效果**：为 Gemini 2.5 Flash 提供 Bluebook 规则后，准确率仅从71%提升至77%，提升幅度有限，显示长上下文学习在处理复杂法律规则时效果不佳。
*   **错误分析**：错误不仅限于小格式问题，包含实质性错误（如错误引用当事人或法院），平均每个错误引用需14个字符编辑修正，表明错误可能影响引用可信度。
*   **实验设置合理性**：数据集覆盖多种任务类型，数量充足（866个），任务分布基于教材具有代表性；测试不同模型和设置（零样本与上下文学习）较为全面，但未探索少样本提示或微调可能带来的改进；总体设置合理，结论支持 LLMs 目前无法满足法律程序任务的高准确性需求。

## Further Thoughts

论文揭示了 LLMs 在程序性任务上的局限性，启发我们重新审视法律AI中‘简单任务’的定义；长上下文学习的低效提示是否可以通过规则摘要或结构化输入优化模型理解；此外，模型依赖记忆而非规则遵循的表现，是否可以通过更多对抗性测试（如合成数据）进一步验证其泛化能力，或许这能为设计更可靠的法律AI工具提供新思路。