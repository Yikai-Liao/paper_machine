---
title: "Do Language Models Think Consistently? A Study of Value Preferences Across Varying Response Lengths"
pubDatetime: 2025-06-03T05:52:03+00:00
slug: "2025-06-value-consistency-llm"
type: "arxiv"
id: "2506.02481"
score: 0.7000521700403703
author: "grok-3-latest"
authors: ["Inderjeet Nair", "Lu Wang"]
tags: ["LLM", "Value Alignment", "Consistency", "Response Length", "Ethical Reasoning"]
institution: ["University of Michigan, Ann Arbor, MI"]
description: "本文通过系统分析揭示了大型语言模型在短篇与长篇回应中价值偏好的不一致性，强调生成模式和应用领域的影响，为未来价值对齐研究提供了新视角。"
---

> **Summary:** 本文通过系统分析揭示了大型语言模型在短篇与长篇回应中价值偏好的不一致性，强调生成模式和应用领域的影响，为未来价值对齐研究提供了新视角。 

> **Keywords:** LLM, Value Alignment, Consistency, Response Length, Ethical Reasoning

**Authors:** Inderjeet Nair, Lu Wang

**Institution(s):** University of Michigan, Ann Arbor, MI


## Problem Background

大型语言模型（LLMs）在伦理价值表达上的评估多基于短篇问卷或心理测试，但现实应用中常需长篇、开放式回应，其价值偏好和风险尚未充分研究。
论文聚焦于解决一个关键问题：从短篇回应中推断的价值偏好是否与长篇回应一致，以及长篇回应在不同长度下的价值表达是否稳定，以避免实际应用中的伦理风险。

## Method

*   **核心框架:** 设计系统性分析方法，提取并比较 LLMs 在短篇和长篇回应中的价值偏好，探索一致性及影响因素。
*   **数据集:** 使用 DAILY DILEMMAS（1360个日常道德困境，包含301个细粒度价值）和 OPINION QA（涵盖健康、犯罪等广泛话题的开放性问题）两个数据集，确保覆盖不同领域和应用场景。
*   **短篇回应价值提取:** 通过提示模型对道德困境做出选择，基于决策推断隐式价值偏好；采用高斯信念分布（Gaussian Belief Distribution）表示偏好强度，并使用 TrueSkill 算法更新偏好参数，量化不确定性。
*   **长篇回应价值提取:** 提示模型生成固定数量（5、10、20个论点）的长篇回应，要求按偏好顺序排列论点；利用 GPT-4o 提取论点及其关联价值，通过论点位置（归一化后取负值）量化偏好。
*   **一致性分析:** 使用皮尔逊相关系数（Pearson Correlation）测量短篇与长篇、不同长篇模式间价值偏好的一致性，并对比对齐（Alignment）前后模型表现。
*   **生成属性分析:** 定义并测量长篇回应中论点的特异性（Specificity，通过路径深度评估论点上下文清晰度）和多样性（Diversity，通过压缩比评估论点在场景中的广泛性），探讨这些属性与价值偏好的关系。

## Experiment

*   **一致性结果:** 在五个 LLMs（Llama3-8B, Gemma2-9B, Mistral-7B, Qwen2-7B, Olmo-7B）上的实验显示，短篇与长篇回应推断的价值偏好相关性较弱（皮尔逊相关系数普遍低于0.25）；长篇回应在不同论点数量（5、10、20）间的一致性也较低，表明价值表达高度依赖生成模式；DAILY DILEMMAS 数据集上的一致性高于 OPINION QA，说明模型在日常道德场景中更稳定。
*   **对齐效果:** 对齐（Alignment，如 DPO 或 RLHF）后一致性有适度提升，但提升不显著，且在不同长篇模式间效果不稳定，表明对齐未根本解决价值不一致问题。
*   **生成属性关联:** 特异性与价值偏好呈负相关（偏好较低的价值论点更具体），多样性与偏好呈正相关（偏好较高的价值在不同场景中表达更广泛）。
*   **实验设置评价:** 实验覆盖多种模型、数据集和生成模式，设置较为全面合理；但局限性在于未测试更大规模模型（>10B 参数），且依赖 GPT-4o 提取论点和价值可能引入偏差。

## Further Thoughts

论文揭示了价值偏好与生成模式（短篇 vs 长篇）和应用领域的强相关性，启发我们思考是否可以通过设计特定提示或微调策略强制模型在不同模式下保持一致性；例如，训练时引入多模式价值对齐任务，或动态调整生成长度以平衡一致性和表达深度；此外，特异性和多样性与偏好的关系提示我们，是否可以通过控制生成属性间接影响价值表达的倾向性。