---
title: "MUG-Eval: A Proxy Evaluation Framework for Multilingual Generation Capabilities in Any Language"
pubDatetime: 2025-05-20T14:14:00+00:00
slug: "2025-05-mugeval-multilingual-eval"
type: "arxiv"
id: "2505.14395"
score: 0.5385678651917176
author: "grok-3-latest"
authors: ["Seyoung Song", "Seogyeong Jeong", "Eunsu Kim", "Jiho Jin", "Dongkwan Kim", "Jamin Shin", "Alice Oh"]
tags: ["LLM", "Multilingual Evaluation", "Generation Capability", "Task Completion", "Low-Resource Language"]
institution: ["KAIST", "Trillion Labs"]
description: ""
---

> **Summary:**  

> **Keywords:** LLM, Multilingual Evaluation, Generation Capability, Task Completion, Low-Resource Language

**Authors:** Seyoung Song, Seogyeong Jeong, Eunsu Kim, Jiho Jin, Dongkwan Kim, Jamin Shin, Alice Oh

**Institution(s):** KAIST, Trillion Labs


## Problem Background

大型语言模型（LLMs）在多语言生成能力评估上面临挑战，尤其是在低资源语言中，缺乏语言特定的自然语言处理工具、参考语料库和人工标注数据；传统评估方法依赖这些资源，而近期‘LLMs-as-judges’方法在高资源语言外的可靠性下降，无法公平评估多语言能力；因此，论文提出 M U G-Eval 框架，旨在以语言无关的方式评估 LLMs 的多语言生成能力。

## Method

* **核心思想：** M U G-Eval 是一个语言无关的评估框架，通过将现有基准转化为对话任务，以任务完成率作为生成能力的代理指标，避免对语言特定工具或人工标注的依赖。
* **任务设计：** 框架包含三个对话任务，基于信息不对称场景，要求两个 LLM 实例在目标语言中有效沟通：
  * **Easy Twenty Questions：** 评估推理和策略性提问能力，基于‘Things’数据集，一个实例（提问者）通过最多 20 个是/否问题猜测隐藏词，另一个实例（回答者）仅回复‘yes’、‘no’或‘maybe’。
  * **MCQ Conversation：** 评估多轮指令遵循能力，基于 Belebele 数据集，提问者通过最多 10 个问题确定多选题答案，回答者根据隐藏段落回复‘yes’、‘no’或‘maybe’。
  * **Code Reconstruction：** 评估代码生成能力，基于 HumanEvalExplain 数据集，一个实例（描述者）用目标语言描述代码，另一个实例（重建者）根据描述重建代码并通过单元测试。
* **评估方式：** 使用任务完成率作为主要指标，通过算法（如字符串匹配、代码测试）客观评分，避免主观判断；同时使用 GlotLID 工具确保输出符合目标语言。
* **优势：** 不依赖语言特定资源或 LLMs-as-judges，适用于高、中、低资源语言，具有高度可扩展性。

## Experiment

* **有效性：** 在 30 种语言（高、中、低资源各 10 种）上测试了 8 个 LLMs（包括 Llama、Qwen、GPT-4o、Gemini 等），M U G-Eval 表现出较强的区分能力，能清晰区分模型和语言间的性能差异；与现有基准（如 Belebele、MultiQ、Global-MMLU）的 Pearson 相关系数达 0.75，验证了框架的可靠性。
* **任务难度分布：** 三个任务难度互补，Easy Twenty Questions 最难（多轮交互易累积错误），Code Reconstruction 最易（单轮交互），MCQ Conversation 居中，共同提升了框架对不同性能范围的区分能力。
* **性能差异：** 高资源语言（如英语）性能显著优于低资源语言，模型规模越大性能越好（如 GPT-4o 优于 GPT-4o-mini）；低资源语言中，英语并非总是最佳替代语言，提示评估需考虑语言相似性。
* **实验设置合理性：** 语言选择覆盖多种语系和书写系统，模型选择兼顾开源与闭源，任务设计基于现有基准，确保了评估的全面性和可比性；但框架未进行广泛的人工验证，可能存在对生成质量细微差异的忽略。

## Further Thoughts

M U G-Eval 通过任务完成率间接评估生成能力的思路非常启发性，未来可以扩展到其他领域（如多模态模型评估），设计类似任务驱动的框架，绕过对主观质量评判的依赖；此外，论文揭示英语并非低资源语言的最佳替代语言，提示我们可以探索基于语言相似性或谱系关系的动态替代策略，以提升评估公平性和准确性。