---
title: "Automated Meta Prompt Engineering for Alignment with the Theory of Mind"
pubDatetime: 2025-05-13T23:42:36+00:00
slug: "2025-05-tom-meta-prompting"
type: "arxiv"
id: "2505.09024"
score: 0.5703201149897325
author: "grok-3-latest"
authors: ["Aaron Baughman", "Rahul Agarwal", "Eduardo Morales", "Gozde Akay"]
tags: ["LLM", "Theory of Mind", "Meta Prompting", "Agentic Architecture", "RLHF"]
institution: ["IBM"]
description: "本文提出一种基于心智理论（ToM）的自动化元提示工程方法，通过多代理架构和强化学习，在 2024 年美国网球公开赛中显著提升了大型语言模型输出与人类期望的对齐程度。"
---

> **Summary:** 本文提出一种基于心智理论（ToM）的自动化元提示工程方法，通过多代理架构和强化学习，在 2024 年美国网球公开赛中显著提升了大型语言模型输出与人类期望的对齐程度。 

> **Keywords:** LLM, Theory of Mind, Meta Prompting, Agentic Architecture, RLHF

**Authors:** Aaron Baughman, Rahul Agarwal, Eduardo Morales, Gozde Akay

**Institution(s):** IBM


## Problem Background

生成式人工智能（Generative AI）在复杂任务中（如体育赛事报道）常无法完全满足人类内容创作者的心理预期，导致内容生成后需要大量人工编辑。
论文旨在解决大型语言模型（LLM）与人类心智理论（Theory of Mind, ToM）的对齐问题，通过自动化方法让模型输出更符合人类期望，减少人工干预并提升内容质量。

## Method

*   **核心思想：** 提出一种自动化元提示工程方法，通过多代理架构和强化学习，动态调整 LLM 的内容生成，使其与人类心智模型对齐。
*   **具体实现：**
    *   **代理架构（Agentic Architecture）：** 使用 IBM Granite 13B Chat 模型生成事实性要点，Llama 3 70B 模型将要点转化为流畅的比赛报道段落，形成多代理协作的内容生成流程。
    *   **LLM 作为评判者（LLM as a Judge, LLMaaJ）：** 另一个 Llama 3 70B 模型根据人类定义的四个维度（事实性 Factualness、新颖性 Novelty、重复性 Repetitiveness、主题相关性 Topic Alignment）对生成内容评分，并与人类编辑后的内容评分对比，计算对齐差异。
    *   **LLM 作为编辑者（LLM as a Editor, LLMaaE）：** 基于评分反馈，IBM Granite 13B Chat V2 模型动态改写提示（Prompt），生成新内容以逼近人类期望。
    *   **强化学习与人类反馈（RLHF）：** 通过迭代优化，利用人类编辑者的反馈调整模型输出，结合几何空间损失函数（基于 Hilbert 空间的面积和距离差异）量化人类与 AI 之间的 ToM 对齐程度。
    *   **元提示（Meta Prompting）：** 自动化生成和优化提示，使模型自适应学习人类偏好，减少对任务特定指令的依赖。
*   **关键点：** 方法不依赖于模型内部调整，而是通过外部提示优化和多代理协作实现对齐，确保灵活性和可扩展性。

## Experiment

*   **有效性：** 在 2024 年美国网球公开赛的实时内容生成中，53.8% 的情况下 AI 内容与人类期望完全对齐（即收敛），平均迭代次数为 4.38 次，编辑者可直接发布内容，表明方法在减少人工干预方面效果显著。
*   **未收敛情况：** 在未收敛的情况下，事实性维度有时比初始状态更差，但新颖性和相关性有所提升，编辑者更倾向于接受未收敛内容而非初始内容，显示方法仍有改进空间。
*   **实验设置：** 实验覆盖 254 场比赛报道，涉及 239 个使用 ToM 对齐优化的后赛总结，数据量大（803,000 用户阅读，14 百万用户访问），场景真实，评价维度包括 3D 和 4D（是否包含重复性），并对比了是否使用自然语言生成模板（NLG）的影响，设置全面合理。
*   **局限性：** 未收敛时部分维度下降可能与优化目标权衡或时间限制（2 分钟或 21 次迭代）有关，实验未深入探讨模型参数或规模对结果的影响。

## Further Thoughts

将人类心智理论（ToM）引入 AI 系统，通过量化人类期望维度指导内容生成，这种思路可扩展至教育、医疗对话等领域增强 AI 社会智能；使用 Hilbert 空间的几何优化方法为多维度对齐问题提供数学化解决方案，未来可用于复杂多目标优化；为每个用户构建个性化 ToM 模型的思路启发我们在更大规模用户群体中实现定制化 AI 服务；多代理协作与 RLHF 的框架可应用于其他动态调整任务，如实时决策系统。