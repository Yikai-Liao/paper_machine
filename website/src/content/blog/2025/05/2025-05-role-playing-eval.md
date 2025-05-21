---
title: "Role-Playing Evaluation for Large Language Models"
pubDatetime: 2025-05-19T14:18:16+00:00
slug: "2025-05-role-playing-eval"
type: "arxiv"
id: "2505.13157"
score: 0.5709259670383529
author: "grok-3-latest"
authors: ["Yassine El Boudouri", "Walter Nuninger", "Julian Alvarez", "Yvan Peter"]
tags: ["LLM", "Role-Playing", "Evaluation Benchmark", "Emotional Understanding", "Consistency"]
institution: ["Univ. Lille", "CNRS", "Centrale Lille", "UMR 9189 CRIStAL"]
description: "本文提出 RPEval 基准，通过单轮交互和自动化评估方法，系统化衡量大型语言模型在角色扮演中的情感理解、决策能力、道德一致性和角色一致性等关键维度。"
---

> **Summary:** 本文提出 RPEval 基准，通过单轮交互和自动化评估方法，系统化衡量大型语言模型在角色扮演中的情感理解、决策能力、道德一致性和角色一致性等关键维度。 

> **Keywords:** LLM, Role-Playing, Evaluation Benchmark, Emotional Understanding, Consistency

**Authors:** Yassine El Boudouri, Walter Nuninger, Julian Alvarez, Yvan Peter

**Institution(s):** Univ. Lille, CNRS, Centrale Lille, UMR 9189 CRIStAL


## Problem Background

大型语言模型（LLMs）在角色扮演（Role-Playing）方面展现出显著能力，但对其能力的评估面临挑战：人工评估成本高且主观性强，基于模型的评估可靠性有限，现有量化基准未能全面覆盖角色扮演的复杂维度。
因此，亟需一个系统化、自动化且可重复的评估框架来衡量 LLMs 在角色扮演中的表现。

## Method

*   **核心思想:** 提出 RPEval 基准，通过单轮交互（Single-Turn Interaction）实现自动化评估，系统化衡量 LLMs 在角色扮演中的能力，聚焦于四个关键维度：情感理解（Emotional Understanding）、决策能力（Decision-Making）、道德一致性（Moral Alignment）和角色一致性（In-Character Consistency）。
*   **具体实现:**
    *   **角色与场景生成:** 开发角色生成工具，结合 GPT-4o 生成了 3125 个多样化角色描述（包括人类和虚构角色），并为每个角色创建多个场景（共 18850 个），覆盖四个评估维度，每个场景包含一个输入信息和预期响应。
    *   **数据标注与筛选:** 通过众包平台收集人类响应（共 48687 条），采用多数投票机制确定预期响应，并设置严格筛选标准（情感理解一致性需 55%，决策/道德一致性需 70%），最终保留 9018 个场景和 3061 个角色。
    *   **评估机制:** 对模型响应进行二元评分（1 或 0），分别衡量四个维度的表现，通过简单条件检查（如情感标签、二元响应、关键词过滤）实现自动化评估，避免多轮对话的复杂性和人工干预。
*   **设计考量:** 优先选择可以通过单轮交互评估的维度，牺牲了对长期对话中角色一致性或情感动态变化的评估，确保成本效率和可重复性。

## Experiment

*   **有效性:** 评估了 GPT-4o、Gemini-1.5-Pro 和 Llama 3.2 1B 三个模型，Gemini-1.5-Pro 表现最佳（平均得分 62.24%），在决策/道德一致性（73.86%）和角色一致性（59.75%）上显著优于其他模型；GPT-4o 平均得分 44.41%，在角色一致性上表现极差（5.81%）；Llama 3.2 1B 平均得分 39.33%，整体表现较弱。
*   **合理性:** 实验设置覆盖了不同规模模型，并通过多次测试（n=6）验证结果稳定性（标准差 0.89%），表明评估结果可靠；但单轮交互的限制可能低估了模型在多轮对话中的潜力，角色一致性评估标准较严格，可能导致部分模型（如 GPT-4o）得分偏低。
*   **局限性:** 无法捕捉长期对话中的角色一致性或情感动态变化，评估维度较为有限，可能未全面反映模型的角色扮演能力。

## Further Thoughts

RPEval 的单轮交互自动化评估方法为资源受限场景下的初步筛选提供了创新思路，启发我们思考如何设计分层评估框架，结合单轮和多轮交互以全面衡量模型能力；此外，将角色扮演能力分解为多个维度的思路为未来扩展评估框架（如加入长期记忆或情感动态调整）提供了清晰方向；众包标注的多样性方法也值得推广至其他主观评估任务中。