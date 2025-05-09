---
title: "Validating the Effectiveness of a Large Language Model-based Approach for Identifying Children's Development across Various Free Play Settings in Kindergarten"
pubDatetime: 2025-05-06T09:40:47+00:00
slug: "2025-05-llm-child-development"
type: "arxiv"
id: "2505.03369"
score: 0.5003095565134422
author: "grok-3-latest"
authors: ["Yuanyuan Yang", "Yuan Shen", "Tianchen Sun", "Yangbin Xie"]
tags: ["LLM", "Learning Analytics", "Early Childhood Education", "Child Development", "Narrative Analysis"]
institution: ["Zhejiang Lab"]
description: "本文提出并验证了一种基于大型语言模型和学习分析的方法，通过分析幼儿园儿童自由玩耍自述，准确评估其多维度发展能力，为教育者提供数据驱动的个性化洞察。"
---

> **Summary:** 本文提出并验证了一种基于大型语言模型和学习分析的方法，通过分析幼儿园儿童自由玩耍自述，准确评估其多维度发展能力，为教育者提供数据驱动的个性化洞察。 

> **Keywords:** LLM, Learning Analytics, Early Childhood Education, Child Development, Narrative Analysis

**Authors:** Yuanyuan Yang, Yuan Shen, Tianchen Sun, Yangbin Xie

**Institution(s):** Zhejiang Lab


## Problem Background

自由玩耍（Free Play）是幼儿教育的重要组成部分，对儿童的认知、社交、情感和运动发展至关重要，但由于其无结构性和自发性，传统评估方法（如教师或家长观察）难以全面捕捉儿童发展表现并提供及时反馈。
作者提出利用大型语言模型（LLMs）结合学习分析（Learning Analytics）分析儿童玩耍自述（Self-Narratives），以解决评估难题，为教育者提供数据驱动的洞察。

## Method

*   **核心思想：** 利用大型语言模型（LLMs）从儿童自由玩耍的自述文本中提取发展能力信息，并通过学习分析量化这些信息，形成可视化评估结果。
*   **具体步骤：**
    *   **数据收集：** 从幼儿园儿童中收集自由玩耍后的自述文本，共 2224 条叙述，涉及 29 名儿童在四个不同玩耍区域的表现，时间跨度为一个学期。
    *   **数据预处理：** 对文本进行校对（如拼写错误修正）并保护隐私（替换儿童姓名以使用唯一标识符）。
    *   **模型整合：** 调用 LLM API（如 Qwen-Max 模型），结合预定义的提示（Prompts）和能力框架（涵盖认知、运动、情感、社交维度），分析自述文本，识别儿童在各能力维度上的表现。提示设计遵循清晰、结构化原则，确保输出与能力类别相关。
    *   **数据格式化：** 将 LLM 生成的内容整理为结构化结果，关联能力类别与具体表现描述，便于后续分析。
    *   **能力评分：** 基于分析结果计算儿童在不同能力维度上的表现分数，公式为某一能力在特定时间段内被推断的次数除以该儿童在同一时间段的总活动记录数，并通过可视化工具（如雷达图）呈现结果。
*   **关键特点：** 该方法自动化处理非结构化文本，结合学习分析提供量化评估，减少人为观察的主观性，同时支持教育者监控儿童发展轨迹。

## Experiment

*   **有效性：** 实验结果表明，LLM-based 方法在认知、运动和社交能力识别上的准确率超过 90%，在情感能力（如情感识别和共情）上的准确率较低（70%-80%），整体遗漏率为 14.1%。
*   **差异性：** 不同玩耍区域对能力发展的影响存在显著差异，例如 Building Blocks Area 对数理几何能力发展贡献最大，Hillside-Zipline Area 对粗大运动能力发展更有效。
*   **合理性：** 实验设置涵盖 29 名儿童、2224 条自述数据，时间跨度为一个学期，并通过 8 名专业人士评估 328 条随机样本，采用语义一致性、能力相关性和遗漏率等多指标验证结果，统计分析（如 ANOVA 和 Kruskal-Wallis H 测试）也较为严谨。
*   **局限性：** 情感能力识别准确率较低，部分能力遗漏率较高，样本量和单一地理位置可能限制结果普适性。

## Further Thoughts

LLM 在教育领域的应用潜力巨大，不仅限于内容生成，还可以通过分析非结构化文本（如儿童自述或学生日记）提供发展评估，未来可结合多模态数据（如语音语调、面部表情）提升情感识别准确性；此外，不同物理环境对儿童发展的独特影响提示我们可以在教育环境设计中优化资源配置，并通过实时反馈系统动态调整教学策略，支持个性化教育。