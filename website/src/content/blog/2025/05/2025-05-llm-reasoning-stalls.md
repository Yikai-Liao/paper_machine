---
title: "Large Language Models' Reasoning Stalls: An Investigation into the Capabilities of Frontier Models"
pubDatetime: 2025-05-26T08:34:07+00:00
slug: "2025-05-llm-reasoning-stalls"
type: "arxiv"
id: "2505.19676"
score: 0.7944029697401623
author: "grok-3-latest"
authors: ["Lachlan McGinness", "Peter Baumgartner"]
tags: ["LLM", "Reasoning", "In-Context Learning", "Automated Theorem Proving", "Chain of Thought"]
institution: ["Australian National University", "CSIRO", "Data61"]
description: "本文通过纵向研究揭示2023-2024年大型语言模型推理能力停滞，指出性能提升主要依赖提示工程而非内在能力改进，并发现自底向上推理策略表现最佳。"
---

> **Summary:** 本文通过纵向研究揭示2023-2024年大型语言模型推理能力停滞，指出性能提升主要依赖提示工程而非内在能力改进，并发现自底向上推理策略表现最佳。 

> **Keywords:** LLM, Reasoning, In-Context Learning, Automated Theorem Proving, Chain of Thought

**Authors:** Lachlan McGinness, Peter Baumgartner

**Institution(s):** Australian National University, CSIRO, Data61


## Problem Background

本文研究大型语言模型（LLMs）在逻辑推理能力上的表现，特别是在使用自动化定理证明（ATP）策略时的能力，旨在探究当前前沿模型的推理能力是否随着时间和更新而显著提升，尤其关注从2023年12月到2024年8月期间推理能力是否停滞，以及模型是否真正通过推理而非猜测或隐藏提示（如 Chain of Thought, CoT）得出正确答案。

## Method

* **研究目标与框架**：通过实证方法评估 LLMs 在逻辑推理任务上的表现，聚焦于是否能有效应用 ATP 推理策略，并分析推理能力随时间的变化。
* **数据集与任务**：选用 PRONTOQA 基准数据集，聚焦于‘Steamroller’逻辑推理问题，要求模型进行演绎推理（modus ponens）；该数据集为问题生成器，避免了数据污染问题。
* **测试模型**：对比2023年12月和2024年8月的前沿模型，包括 GPT-3.5 Turbo、GPT-4、GPT-4o、Google Gemini-Pro、Claude 3 Opus 和 Llama 3.1 405B，以进行纵向能力评估。
* **推理策略设计**：通过上下文学习（in-context learning）教导模型使用三种 ATP 推理策略：
  * 自底向上（Bottom Up，即前向链推理）：从事实出发，逐步应用规则推导出结论，直至回答查询。
  * 自顶向下（Top Down，即后向链推理）：从查询出发，递归推导出子目标，直至子目标可被事实证明或证伪。
  * 魔法集变换（Magic Set Transformation）：结合自顶向下确定相关规则和事实，再用自底向上推理，减少搜索空间。
* **提示技术**：测试多种提示方式，包括普通提示（Normal）、零样本 CoT（Zero-shot CoT）、单样本 CoT（One-shot CoT）以及基于 ATP 策略的定制提示，以对比不同引导方式对推理表现的影响。
* **评估手段**：
  * 准确性评估：统计模型在不同推理步数（1-hop, 2-hop, 3-hop）下的答案正确率。
  * 推理过程分析：使用 SpaCy 工具（自然语言处理工具）解析模型输出，评估推理步骤的正确性及对指定推理策略的忠实度（faithfulness）。
  * 统计分析：通过平均值、范围和误差条量化结果不确定性，确保结果稳健。
* **实验执行**：针对每个模型、提示技术和推理步数组合进行多次调用（共计1800次调用/模型），并自动化验证推理过程，以避免人工验证的不可行性。

## Experiment

* **准确性结果**：从2023年12月到2024年8月，前沿模型在零样本情况下的准确性有所提升，但其他实验条件下的表现与 GPT-4 相比无显著改进；例如，GPT-4 在多个推理策略下仍保持领先或接近2024年新模型的水平。
* **推理能力停滞**：通过完成 token 数量分析，发现2024年模型在普通提示下自动使用 CoT 推理（可能是训练或隐藏提示所致），这解释了零样本性能提升，但表明内在推理能力未有实质性进步。
* **推理策略表现**：模型在自底向上（Bottom Up）推理策略下表现最佳，准确性和对策略的忠实度最高，而在魔法集变换（Magic Set Transformation）等复杂策略下表现较差。
* **相关性分析**：正确推理与正确答案之间存在弱正相关，表明模型有时可能通过猜测而非严谨推理得出正确答案。
* **实验设置评价**：实验设计较为全面，涵盖多个模型、提示技术和推理步数，通过统计方法量化不确定性，避免单纯追求准确性的偏差；使用 PRONTOQA 数据集避免过拟合问题，增强结果可信度；但未考虑计算成本（如 FLOPs 或延迟），且测试问题为简单玩具问题，未涉及更复杂 ATP 基准。

## Further Thoughts

论文揭示了 LLMs 推理能力提升的瓶颈可能不在于模型规模或数据量，而在于缺乏创新的推理增强技术，这启发我们探索神经-符号（neuro-symbolic）方法，将 LLMs 与传统 ATP 结合以提升推理可信度；此外，隐藏提示（如自动 CoT）可能限制新提示技术的研究，提示我们需要更透明的模型设计来支持推理策略创新；最后，自底向上推理策略的优越表现可能与 LLMs 自回归预测机制契合，未来提示工程或模型架构设计可优先考虑与模型天然契合的推理模式。