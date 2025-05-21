---
title: "Leveraging LLM Inconsistency to Boost Pass@k Performance"
pubDatetime: 2025-05-19T10:22:04+00:00
slug: "2025-05-llm-inconsistency-variator"
type: "arxiv"
id: "2505.12938"
score: 0.6452982670191609
author: "grok-3-latest"
authors: ["Uri Dalal", "Meirav Segal", "Zvika Ben-Haim", "Dan Lahav", "Omer Nevo"]
tags: ["LLM", "Inconsistency Effect", "Pass@k Performance", "Variant Generation", "Reasoning"]
institution: ["Pattern Labs"]
description: "本文提出 Variator 代理，通过生成任务等价变体利用大型语言模型的不一致性，显著提升 Pass@k 性能，并在编程和网络安全任务上通过理论和实验验证了其有效性。"
---

> **Summary:** 本文提出 Variator 代理，通过生成任务等价变体利用大型语言模型的不一致性，显著提升 Pass@k 性能，并在编程和网络安全任务上通过理论和实验验证了其有效性。 

> **Keywords:** LLM, Inconsistency Effect, Pass@k Performance, Variant Generation, Reasoning

**Authors:** Uri Dalal, Meirav Segal, Zvika Ben-Haim, Dan Lahav, Omer Nevo

**Institution(s):** Pattern Labs


## Problem Background

大型语言模型（LLMs）在多种任务中表现出色，但对输入提示的微小变化（如同义词替换或背景故事调整）表现出显著的性能不一致性（Inconsistency Effect），这通常被视为可靠性问题。
本文的出发点是将这种不一致性从缺陷转化为优势，特别是在 Pass@k 指标下提升模型性能，Pass@k 允许模型提交 k 个候选解决方案，只要其中一个正确即视为成功，这种场景在编程和网络安全等领域尤为适用。
关键问题是如何利用模型对不同输入变体的不同响应，增加至少一个正确解决方案的概率，从而提高整体性能。

## Method

*   **核心思想:** 提出一种名为‘Variator’代理的方法，通过生成任务的多个等价变体（variants），利用模型的不一致性来提升 Pass@k 性能。
*   **具体实现步骤:**
    *   **变体生成:** 使用 LLM 自动生成 k 个等价的任务变体，这些变体在语义上等价（即解决方案可以互换），但在表述、背景故事、术语或格式上有所不同。生成过程通过结构化提示指导模型修改任务描述、背景、数学符号等，同时保持输入输出格式一致，确保等价性。
    *   **解决方案生成:** 对每个变体，模型生成一个候选解决方案，并提交这些解决方案进行验证。相比之下，基线方法‘Repeater’是对原始任务生成 k 个解决方案。
    *   **理论依据:** 通过概率模型分析，即使变体的平均成功率与原始任务相同，Pass@k 指标会放大某些变体上的性能提升，尤其是在困难任务上，因为低成功率下的小幅提升对 Pass@k 的贡献远大于高成功率下的小幅下降。
*   **关键特点:** 方法是任务无关的（task-agnostic），适用于自由格式输入，且不需要人工干预变体生成，具备较强的通用性。

## Experiment

*   **有效性:** 实验在编程（APPS 数据集）和网络安全（CTF 挑战）两个领域，使用 OpenAI o3-mini 和 Claude 3.7 Sonnet 两个前沿推理模型进行验证。结果显示，Variator 在公开数据集上 k≥5（o3-mini）和 k≥10（Claude 3.7）时优于基线 Repeater，在私有数据集上提升更显著（如 Claude 3.7 在 k=10 时提升约 4 个百分点），表明方法成功利用了不一致性。
*   **实验设置合理性:** 实验涵盖公开和私有数据集，以消除模型对公开数据的记忆效应（memorization effect）；变体等价性通过专家手动验证（约 6% 非等价变体被剔除）；实验考虑了不同 k 值的影响，验证了理论预测；统计显著性测试（p-value < 5×10^-4）确认了不一致性效应的存在。
*   **局限性与开销:** 变体生成可能偶尔产生非等价变体，影响性能；计算成本增加（额外生成变体需更多 token 和时间）；实验规模受限于计算资源，仅在 APPS 子集上进行。

## Further Thoughts

论文启发我们将模型的缺陷（如不一致性）转化为性能提升的工具，这种逆向思维可以扩展到其他领域，例如利用模型的偏见或噪声来增强多样性生成或对抗性测试。此外，作者关于模型响应路径的假设（响应生成如决策树遍历，早期决策影响后续成功率）为理解和改进模型推理过程提供了新视角，或许可以通过干预早期 token 生成来引导模型选择更优路径。