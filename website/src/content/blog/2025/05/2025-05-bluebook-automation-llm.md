---
title: "Bye-bye, Bluebook? Automating Legal Procedure with Large Language Models"
pubDatetime: 2025-05-05T16:18:07+00:00
slug: "2025-05-bluebook-automation-llm"
type: "arxiv"
id: "2505.02763"
score: 0.5863059716538778
author: "grok-3-latest"
authors: ["Matthew Dahl"]
tags: ["LLM", "Legal Citation", "Procedural Rules", "In-Context Learning", "Rule Following"]
institution: ["Yale Law School"]
description: "本文通过原创数据集和实证评估，揭示了大型语言模型在《Bluebook》法律引用格式化任务上的局限性（准确率仅 69%-74%），为 LLMs 在法律程序性任务中的应用提供了重要基准和研究方向。"
---

> **Summary:** 本文通过原创数据集和实证评估，揭示了大型语言模型在《Bluebook》法律引用格式化任务上的局限性（准确率仅 69%-74%），为 LLMs 在法律程序性任务中的应用提供了重要基准和研究方向。 

> **Keywords:** LLM, Legal Citation, Procedural Rules, In-Context Learning, Rule Following

**Authors:** Matthew Dahl

**Institution(s):** Yale Law School


## Problem Background

法律实践要求严格遵守程序性规则，而《Bluebook: A Uniform System of Citation》作为美国法律引用的复杂标准，其格式化任务耗费大量人力。
论文探讨大型语言模型（LLMs）是否能自动化此类程序性任务，以减轻法律从业者负担，并作为评估 LLMs 在更广泛法律程序性规则遵循能力的第一步，解决现有研究对程序性任务关注不足的问题。

## Method

*   **数据集构建**：作者创建了一个包含 866 个《Bluebook》格式化任务的原创数据集，任务类型包括案例法、成文法和其他法律资源引用，分为 cloze 填空和开放式生成两种格式，基于专家验证的法律写作资源，确保答案准确性。
*   **零样本测试**：选择了五个旗舰 LLMs（OpenAI 的 GPT 4.1、Anthropic 的 Claude 3.5 Sonnet、Google 的 Gemini 2.5 Flash、Meta 的 Llama 3.1 405B、DeepSeek 的 V3 0324），在无额外训练或提示的零样本设置下，测试其生成符合《Bluebook》规则引用的能力。
*   **上下文学习测试**：针对 Gemini 2.5 Flash（因其长上下文能力突出），提供《Bluebook》规则（通过公共领域 Indigo Book，约 90k tokens）作为上下文，评估其性能提升，探索 LLMs 是否能通过直接规则输入提高程序性任务表现。
*   **评估方法**：采用精确字符串匹配评估 LLMs 输出与专家答案的一致性，忽略次要样式错误（如斜体），但也在附录中报告严格样式规则下的结果；此外，通过对比真实案例与合成数据的表现，分析模型是否依赖记忆而非规则理解。
*   **目标与设计**：方法旨在隔离 LLMs 的程序性规则遵循能力，排除对法律实质性内容的依赖，通过多样化任务和模型设置，全面测试其在复杂规则环境下的能力边界。

## Experiment

*   **零样本性能**：五个旗舰 LLMs 在零样本设置下的《Bluebook》任务准确率仅为 69%-74%，GPT 4.1 表现最佳（74%），Llama 3.1 405B 最差（69%），表明 LLMs 无法完全替代人工完成高精度法律格式化任务。
*   **任务类型差异**：模型在案例法任务上表现较好（平均 83%），但部分成功源于对常见案例引用的记忆，而非规则理解；在成文法（平均 41%）和其他任务（平均 34%）上表现较差，尤其是电子版法规引用（仅 10%）。
*   **上下文学习效果**：为 Gemini 2.5 Flash 提供规则上下文后，准确率仅从 71% 提升至 77%，表明长上下文学习对复杂程序性规则的帮助有限。
*   **错误分析**：LLMs 的错误不仅限于小格式问题，还包括实质性错误（如误报当事人或法院），可能影响法律文件的可信度和可查找性。
*   **实验设置评价**：实验设计全面，涵盖多种任务类型和模型，数据集基于专家资源，具有代表性；但局限性包括仅关注《Bluebook》从业者规则，未测试来源验证任务，未尝试微调或少样本提示等可能提升性能的方法。

## Further Thoughts

论文揭示了 LLMs 在长上下文学习中的局限性，提示未来可探索结构化规则表示（如模块化指令）而非单纯依赖上下文输入；同时，区分记忆与规则理解的实验设计启发我们设计更多对抗性测试，确保模型真正掌握规则；此外，程序性任务的机械特性表明结合领域微调或规则引擎可能提高精度，特别是在法律领域对错误容忍度极低的情况下。