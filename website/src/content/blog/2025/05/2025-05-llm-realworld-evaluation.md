---
title: "Evaluating LLM Metrics Through Real-World Capabilities"
pubDatetime: 2025-05-13T06:02:37+00:00
slug: "2025-05-llm-realworld-evaluation"
type: "arxiv"
id: "2505.08253"
score: 0.521529476541091
author: "grok-3-latest"
authors: ["Justin K. Miller", "Wenjia Tang"]
tags: ["LLM", "User-Centric Evaluation", "Benchmark Analysis", "Real-World Utility", "Conversational AI"]
institution: ["University of Sydney"]
description: "本文通过分析现实世界中 AI 使用模式，揭示现有基准与用户需求的脱节，提出以用户为中心的能力评估框架，为更贴近实际应用的模型评估提供了新方向。"
---

> **Summary:** 本文通过分析现实世界中 AI 使用模式，揭示现有基准与用户需求的脱节，提出以用户为中心的能力评估框架，为更贴近实际应用的模型评估提供了新方向。 

> **Keywords:** LLM, User-Centric Evaluation, Benchmark Analysis, Real-World Utility, Conversational AI

**Authors:** Justin K. Miller, Wenjia Tang

**Institution(s):** University of Sydney


## Problem Background

当前对大型语言模型（LLMs）的评估主要依赖于抽象的智能基准测试（如 MMLU、AIME），这些测试聚焦于编码、事实回忆或学术问题解决能力，但与现实世界中用户如何使用 AI（如写作辅助、总结、技术支持）的实际需求存在显著脱节。
论文旨在探究现有基准是否能真实反映 LLMs 的实用性，并通过识别用户核心使用能力，解决评估体系与实际应用不匹配的关键问题。

## Method

* **数据驱动的用户行为分析**：利用丹麦工人大规模调查数据（超过 18,000 名工人）和 Anthropic 的 Claude.ai 使用日志（超过 400 万条提示），识别 AI 在职业任务中的实际应用模式，捕捉用户如何在日常工作中依赖 AI。
* **核心能力分类**：通过定性主题分析，将 AI 使用场景归纳为六个核心能力：Summarization（总结）、Technical Assistance（技术支持）、Reviewing Work（工作审查）、Data Structuring（数据结构化）、Generation（生成）、Information Retrieval（信息检索），为评估提供用户导向的框架。
* **基准评估与标准设计**：基于五个以用户为中心的目标标准（Coherence 一致性、Accuracy 准确性、Clarity 清晰度、Relevance 相关性、Efficiency 效率），系统评估现有基准对六个能力的覆盖情况，分析其设计是否贴近真实交互需求。
* **模型性能比较**：针对四个有对应基准的能力（Summarization、Technical Assistance、Information Retrieval、Generation），选择最符合用户标准的基准（如 WebDev Arena、SimpleQA），并比较主流模型（如 Google Gemini、OpenAI GPT）的表现。
* **方法特点**：强调从用户实际需求出发，结合实证数据与理论标准，重新审视评估体系，而非单纯依赖技术指标或抽象任务。

## Experiment

* **能力分布结果**：通过 Claude.ai 提示数据分析，Technical Assistance（65.1%）和 Reviewing Work（58.9%）是用户最常使用的能力，表明 AI 在问题解决和评估任务中应用最广，而 Data Structuring（4.0%）使用较少。
* **基准覆盖不足**：现有基准仅覆盖 Summarization、Technical Assistance、Information Retrieval 和 Generation 四个能力，Reviewing Work 和 Data Structuring 完全缺乏对应测试工具，暴露评估体系的盲区。
* **模型性能表现**：Google Gemini 2.5 在有基准的四个能力中表现最佳，在 Summarization（89.1%）、Generation（Elo 1458）和 Technical Assistance（Elo 1420）中排名第一，显示出跨任务的强大适应性。
* **实验设置评价**：实验结合大规模用户数据和多维度评估标准，设置较为全面，揭示了基准与实际使用的脱节问题；但对 Reviewing Work 和 Data Structuring 缺乏直接测试，且效率维度未被量化，部分基准（如 WebDev Arena）在领域覆盖（如后端开发）上有限，实验仍有改进空间。
* **方法提升效果**：相比传统抽象智能测试，论文提出的用户导向分析方法显著提升了对实用性评估的关注度，为未来基准设计提供了清晰方向。

## Further Thoughts

论文提出的以用户为中心的评估标准（尤其是 Efficiency 效率维度）启发了我，未来可以通过量化用户任务完成时间或认知负荷来评估 AI 的实际效率；此外，多轮交互（multi-turn interaction）的重视提示我们，评估应模拟真实对话场景，测试模型在上下文适应和迭代改进中的表现；另一个想法是针对不同行业（如医疗、法律）设计定制化基准，捕捉领域特定需求和伦理考量。