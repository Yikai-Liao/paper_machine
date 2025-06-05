---
title: "An Empirical Study of Group Conformity in Multi-Agent Systems"
pubDatetime: 2025-06-02T05:22:29+00:00
slug: "2025-06-group-conformity-llm"
type: "arxiv"
id: "2506.01332"
score: 0.44789293042482875
author: "grok-3-latest"
authors: ["Min Choi", "Keonwoo Kim", "Sungwon Chae", "Sangyeob Baek"]
tags: ["LLM", "Multi-Agent System", "Group Conformity", "Bias Propagation", "Opinion Dynamics"]
institution: ["Kim & Chang AI&IT System Center"]
description: "本文通过多智能体辩论模拟，揭示了大型语言模型智能体在群体一致性中受到数量优势和智能水平的影响，警示了AI驱动舆论动态中的偏见传播风险。"
---

> **Summary:** 本文通过多智能体辩论模拟，揭示了大型语言模型智能体在群体一致性中受到数量优势和智能水平的影响，警示了AI驱动舆论动态中的偏见传播风险。 

> **Keywords:** LLM, Multi-Agent System, Group Conformity, Bias Propagation, Opinion Dynamics

**Authors:** Min Choi, Keonwoo Kim, Sungwon Chae, Sangyeob Baek

**Institution(s):** Kim & Chang AI&IT System Center


## Problem Background

随着大型语言模型（LLMs）在多智能体系统中的应用日益广泛，其在模拟人类交互和决策时展现出接近人类的推理能力。然而，现有研究多集中于LLMs在性别、种族等显性偏见上的表现，而对多智能体交互中关于社会争议性议题的偏见生成与传播研究不足。本文聚焦于LLM智能体在辩论中如何影响公共舆论，尤其是在群体一致性（Group Conformity）方面的表现，试图揭示中立智能体是否会因多数派意见或更高智能水平的影响而表现出从众倾向，并探讨这种倾向是否会放大偏见风险。

## Method

* **核心设计：** 构建一个多智能体辩论系统，包含支持方（Proponent）、反对方（Opponent）和中立方（Neutral Agent），通过模拟辩论来研究一致性行为。
* **变量控制：** 选择五个社会争议性议题（如全民基本收入、移民政策），并通过控制智能体数量（多数 vs 少数）和智能水平（基于模型参数规模，如大型模型 vs 小型模型）来测试中立智能体的从众倾向。
* **实验设置：** 实验分为两部分：Experiment A 研究数量和智能水平对一致性的影响，设置不同场景（如支持方数量占优或智能水平较高）；Experiment B 探讨多数-少数比例的极端变化（如 1:2 到 1:8）对一致性的影响。
* **量化指标：** 定义一致性率（Conformity Rate, CR，表示中立智能体支持某一方的轮次比例）和完全一致性比例（Full Conformity Ratio, FCR，表示中立智能体在所有轮次中始终支持一方的讨论比例）来衡量从众程度。
* **统计分析：** 使用卡方检验（Chi-Square Test）和双因素方差分析（Two-Way ANOVA）验证结果显著性，并通过稳健性测试（如 Welch’s ANOVA）确保数据可靠性。
* **模型选择：** 使用来自不同提供商的模型（如 GPT、Claude、Qwen），以确保结果的泛化性，并通过预测试评估中立智能体的初始偏见，采用成对比较设计消除其影响。

## Experiment

* **有效性：** 实验结果表明中立智能体表现出显著的群体一致性，倾向于支持数量占优的群体或智能水平更高的智能体。例如，在数量占优的场景下，一致性率（CR）显著提高（如 ID a 为 63.53%，ID b 为 39.40%）；在智能水平占优的场景下，效果更为明显（如 ID e 的 CR 高达 74.33%，远超 ID f 的 39.83%）。
* **比例影响：** Experiment B 显示，随着多数-少数比例增加（如从 1:2 到 1:8），一致性率持续上升，尤其在较低智能水平的模型（如 GPT-3.5-turbo）中更为显著。
* **设置合理性：** 实验覆盖多个模型、五个议题、超过 2500 次模拟，并通过统计检验（p < 0.001）确认结果显著性；同时通过成对比较设计消除中立智能体初始偏见的影响，增强了结果的可信度。
* **局限性：** 实验仅使用英语进行辩论，可能引入文化和语言偏见；未考虑人类干预对一致性动态的影响；议题范围有限，未完全覆盖社会问题的多样性。

## Further Thoughts

本文揭示了智能水平在多智能体系统中的主导作用，即使数量处于劣势，高智能水平的智能体也能显著影响中立智能体的立场，这提示我们在设计AI系统时需警惕‘智能权威’可能导致的偏见放大，尤其是在公共舆论形成中，少数高性能模型可能不成比例地主导话语权。此外，多智能体系统作为研究人类社会行为的模拟工具（例如群体极化和沉默螺旋现象），为跨学科研究提供了新视角。进一步思考，是否可以通过调整智能体之间的‘智能差距’或引入‘反权威’机制（如随机化智能体权重）来缓解一致性偏见？是否可以在多智能体系统中引入‘文化背景’变量，模拟不同文化下的从众行为差异？