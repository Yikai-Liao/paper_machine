---
title: "Performance of LLMs on Stochastic Modeling Operations Research Problems: From Theory to Practice"
pubDatetime: 2025-06-30T14:54:15+00:00
slug: "2025-06-llm-stochastic-modeling"
type: "arxiv"
id: "2506.23924"
score: 0.5382416699559893
author: "grok-3-latest"
authors: ["Akshit Kumar", "Tianyi Peng", "Yuhang Wu", "Assaf Zeevi"]
tags: ["LLM", "Stochastic Modeling", "Simulation Optimization", "Reasoning", "Automation"]
institution: ["Columbia Business School, Columbia University"]
description: "本文首次系统评估了大型语言模型在随机建模问题上的能力，从理论到实践展示了其潜力与局限，为构建运营研究智能代理奠定了基础。"
---

> **Summary:** 本文首次系统评估了大型语言模型在随机建模问题上的能力，从理论到实践展示了其潜力与局限，为构建运营研究智能代理奠定了基础。 

> **Keywords:** LLM, Stochastic Modeling, Simulation Optimization, Reasoning, Automation

**Authors:** Akshit Kumar, Tianyi Peng, Yuhang Wu, Assaf Zeevi

**Institution(s):** Columbia Business School, Columbia University


## Problem Background

大型语言模型（LLMs）在多个领域展现了专家级能力，但其在运营研究（Operations Research, OR）中的随机建模（Stochastic Modeling）问题上的表现尚未被充分探索。
论文旨在评估 LLMs 是否能够解决涉及不确定性下的决策制定问题，从理论分析到实际应用，助力 OR 研究者并推动自动化管道的实现。

## Method

*   **数据集构建**：手动收集了三类随机建模问题数据集，包括 71 个研究生水平家庭作业问题（涵盖概率论、随机过程和随机建模）、8 个博士资格考试问题，以及基于 `SimOpt` 库的 6 个仿真优化问题，旨在从课堂到现实世界全面测试 LLMs 的能力。
*   **模型选择与测试**：评估了多个先进 LLMs（如 GPT-4o, o1, o3-mini, Claude 3.5 Sonnet, Llama 3.3 70B, DeepSeek-R1），通过 API 调用和统一提示模板测试其原生能力，避免使用复杂提示工程或链式思维方法，以确保评估的公平性和模型内在能力的体现。
*   **评估策略**：
    *   对于家庭作业问题，采用‘LLM 作为评判者’（LLM-as-a-Judge）方法，使用 GPT-4o 评分，并通过多次评分取平均值提高可靠性。
    *   对于资格考试问题，结合人工评分与 GPT-4o 评分，验证评分一致性。
    *   对于仿真优化问题，执行 LLMs 生成的 Python 代码，比较其结果与 `SimOpt` 库中基准算法（如 RandomSearch, ASTRO-DF）的表现，分析模型在实际决策中的有效性。
*   **核心设计理念**：通过多层次、多场景的测试，评估 LLMs 在理论理解、分析推理和实际问题解决中的综合能力，关注从‘抽象与建模’到‘分析与优化’的 OR 管道阶段。

## Experiment

*   **家庭作业问题**：所有 LLMs 表现优异，平均分远高于及格线（60%），o1 和 o3-mini 表现最佳（总分分别为 94.65 和 96.05），随机建模问题因开放性较高而最具挑战性，评分标准误差较小表明模型稳定性高。
*   **资格考试问题**：o1, o3-mini 和 Claude 3.5 Sonnet 平均分分别为 94.5, 95.75 和 84.63（人工评分），与博士候选人水平相当，人工与 GPT-4o 评分相关系数为 0.77，验证了‘LLM 作为评判者’的可靠性，但答案有时缺乏严谨性。
*   **仿真优化问题**：Claude 3.5 Sonnet 在 5/6 个问题上接近最优解，表现最佳，而 GPT-4o 和 o1 在部分问题上未能生成合理解决方案，表明理论表现与实际应用存在差距，自动化管道仍需改进。
*   **实验设置评价**：实验覆盖理论到实践多个维度，数据来源具有代表性，方法提升在理论问题上显著，但在实际问题上不稳定，实验设计合理但需更多关注现实场景的复杂性。

## Further Thoughts

LLMs 在随机建模问题上的潜力表明，构建领域特定的 OR 智能代理可能极大提升研究效率，未来可探索结合领域知识库或代码验证模块，弥合理论与实践差距；此外，‘LLM 作为评判者’方法的可靠性启发我们思考如何将其扩展到更复杂的开放性问题评估中。