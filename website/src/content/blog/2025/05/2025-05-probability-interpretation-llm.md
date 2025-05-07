---
title: "What do Language Model Probabilities Represent? From Distribution Estimation to Response Prediction"
pubDatetime: 2025-05-04T11:46:48+00:00
slug: "2025-05-probability-interpretation-llm"
type: "arxiv"
id: "2505.02072"
score: 0.7871262245059988
author: "grok-3-latest"
authors: ["Eitan Wagner", "Omri Abend"]
tags: ["LLM", "Distribution Estimation", "Response Prediction", "Training Stages", "Inference Strategies"]
institution: ["Hebrew University of Jerusalem"]
description: "本文通过理论框架区分了语言模型输出概率的三种解释（补全分布、响应分布、事件分布），揭示了现有研究中的混淆，为LLM的概率解释和应用提供了理论指导。"
---

> **Summary:** 本文通过理论框架区分了语言模型输出概率的三种解释（补全分布、响应分布、事件分布），揭示了现有研究中的混淆，为LLM的概率解释和应用提供了理论指导。 

> **Keywords:** LLM, Distribution Estimation, Response Prediction, Training Stages, Inference Strategies

**Authors:** Eitan Wagner, Omri Abend

**Institution(s):** Hebrew University of Jerusalem


## Problem Background

随着大型语言模型（LLMs）的兴起，语言建模从传统的对有限长度字符串的分布估计转向了通用文本输入输出的响应预测，导致输出概率的预期分布因任务目标不同而产生冲突；本文旨在分析这种转变对概率解释的影响，并解决如何正确理解和应用这些分布以避免误解的问题。

## Method

* **核心思想：** 通过理论分析和形式化定义，区分语言模型输出概率的三种不同解释，即源分布估计（反映训练数据分布）、目标分布估计（反映真实世界事件分布）和响应预测（追求输出准确性）。
* **具体分析：** 
  - 形式化定义了三种任务及其对应的理想输出分布，探讨了它们之间的差异和冲突。
  - 分析了LLMs的训练阶段，包括预训练（Pre-Training，分布估计）、监督微调（Supervised Fine-Tuning, SFT，优化响应质量）和偏好调整（如RLHF，基于人类反馈调整输出），指出不同阶段如何影响输出分布。
  - 研究了推理策略对分布的影响，包括朴素补全（Naïve Completion，基于语言模型直接生成）、零样本指令（Zero-Shot Instruction，利用指令提示生成响应）、少样本学习（Few-Shot Learning，通过上下文示例引导输出）和显式概率报告（Explicit Probability Report，模型直接输出概率值）。
  - 通过案例（如抛硬币概率）和文献综述，揭示了现有NLP研究中对三种分布的混淆及其导致的误解。
* **特点：** 本文不涉及具体算法或模型改进，而是提供一个理论框架，用于指导对LLM输出概率的解释和应用。

## Experiment

* **实验设置：** 本文未进行具体实验，而是以理论分析和文献综述为主，通过引用现有研究（如Hu and Levy, 2023; Yona et al., 2024）支持其论点。
* **效果评估：** 由于缺乏实验数据，无法直接评估方法的有效性或性能提升，但论文指出了许多NLP工作中对概率分布的误解（如假设补全分布与响应分布相同），并通过逻辑推演说明了区分三种分布的必要性。
* **局限性：** 理论分析虽全面，但缺乏实证验证，适用性需后续研究进一步确认。

## Further Thoughts

论文提出的输出概率三种解释（补全分布、响应分布、事件分布）的区分，启发我们在设计和评估LLM时需明确任务目标，例如在问答系统中是追求准确性（响应预测）还是反映真实分布（目标分布估计）；此外，显式概率报告作为获取事件概率的潜在一致方法，提示未来可探索更有效的概率表达或校准技术，以克服表达复杂分布的局限性。