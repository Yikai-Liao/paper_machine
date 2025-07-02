---
title: "Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games"
pubDatetime: 2025-06-29T15:02:47+00:00
slug: "2025-06-reasoning-cooperation-dilemma"
type: "arxiv"
id: "2506.23276"
score: 0.7550102045384395
author: "grok-3-latest"
authors: ["David Guzman Piedrahita", "Yongjin Yang", "Mrinmaya Sachan", "Giorgia Ramponi", "Bernhard Schölkopf", "Zhijing Jin"]
tags: ["LLM", "Cooperation", "Social Dilemma", "Sanctioning", "Reasoning"]
institution: ["University of Zurich", "KAIST AI", "ETH Zürich", "MPI for Intelligent Systems, Tuebingen, Germany", "University of Toronto", "Vector Institute"]
description: "本文通过公共物品博弈实验揭示了推理型大型语言模型在社会困境中更倾向于搭便车行为，而传统模型表现出更强的合作能力，挑战了推理能力普遍提升性能的假设。"
---

> **Summary:** 本文通过公共物品博弈实验揭示了推理型大型语言模型在社会困境中更倾向于搭便车行为，而传统模型表现出更强的合作能力，挑战了推理能力普遍提升性能的假设。 

> **Keywords:** LLM, Cooperation, Social Dilemma, Sanctioning, Reasoning

**Authors:** David Guzman Piedrahita, Yongjin Yang, Mrinmaya Sachan, Giorgia Ramponi, Bernhard Schölkopf, Zhijing Jin

**Institution(s):** University of Zurich, KAIST AI, ETH Zürich, MPI for Intelligent Systems, Tuebingen, Germany, University of Toronto, Vector Institute


## Problem Background

随着大型语言模型（LLMs）越来越多地被部署为自主代理，理解它们在多代理系统中的合作行为变得至关重要，尤其是在社会困境中个体利益与集体利益冲突的场景下。
论文聚焦于公共物品博弈中的‘搭便车问题’，研究 LLMs 是否能在类似人类的社会环境中展现合作行为，特别是是否愿意通过‘昂贵的制裁’（即花费自身资源来激励合作或惩罚背叛）来维持集体福祉。
这一问题不仅关乎模型的安全部署和稳健性，也对设计多代理系统的治理机制具有重要意义。

## Method

*   **实验范式：** 基于行为经济学的公共物品博弈，设计了一个包含 7 个 LLM 代理、持续 15 轮的模拟环境，代理需决定是否加入有制裁机制的机构（Sanctioning Institution, SI）或无制裁机制的机构（Sanction-Free Institution, SFI），并决定贡献多少资源到公共物品中；SI 中的代理还可使用额外资源对其他成员进行奖励或惩罚。
*   **模型选择：** 测试两类 LLMs，包括传统模型（如 DeepSeek-V3、GPT-4o、GPT-4o-mini、Llama-3.3-70B）和专注于推理的模型（如 o1-mini、o1-preview、o3-mini 不同推理设置），以对比不同架构和训练范式对合作行为的影响。
*   **提示与决策：** 通过结构化提示向代理提供博弈规则、个人历史记录（过去五轮的决策和结果）以及其他代理的匿名数据，要求代理在每轮做出制度选择、贡献决策和制裁决策，并提供决策理由；提示中不包含明确的合作目标，以观察模型的内在合作倾向。
*   **数据收集与分析：** 记录代理的所有决策和推理文本，使用 GPT-4o 对推理内容进行分类（分为经济推理、社会合作、风险管理和控制策略等类别），并通过统计分析（如分层自举法）比较不同行为类型的推理模式。
*   **核心目标：** 通过模拟人类在社会困境中的行为，揭示 LLMs 的合作能力及其推理过程对合作或背叛行为的影响。

## Experiment

*   **合作效果：** 传统 LLMs 在合作水平上显著优于推理型 LLMs，例如 Llama-3.3-70B 的平均贡献为 18.71 令牌（接近人类水平 18.3 令牌），而 o1-mini 仅为 5.39 令牌，且搭便车比例高达 69.33%。
*   **制度选择：** 传统模型几乎一致选择有制裁机制的机构（SI 参与率高达 99.62%），而推理模型参与率较低（o1-mini 仅 28%），表明推理模型更倾向于避免制裁环境。
*   **收益对比：** 传统模型的平均每轮收益和累计收益更高，例如 GPT-4o 每轮收益 48.07 令牌，而 o1-mini 仅 39.83 令牌，显示合作行为的直接经济效益。
*   **行为模式：** 实验识别出四种行为类型：持续高合作（传统模型）、逐渐背叛（推理模型如 o1-mini）、无变化（固定策略，如 o3-mini-low/med）和不稳定（波动行为，如 o1-preview），揭示了模型在合作上的多样性。
*   **实验设置评价：** 实验设计较为全面，涵盖多种模型和多次运行（部分高成本模型除外），通过历史数据和匿名信息模拟现实中的信息不对称，博弈参数与人类实验一致；但由于部分模型仅运行一次，数据可能有偏差，且 API 模型的随机性和更新可能影响结果的可重复性。

## Further Thoughts

论文揭示了一个重要的启发：推理能力的增强并不必然促进合作行为，反而可能因过度优化个体利益而导致搭便车行为，这挑战了‘更强推理能力等于更好整体性能’的普遍假设，提示我们在设计多代理系统时需针对性地培养亲社会行为，而非单纯依赖推理能力；此外，LLMs 倾向于使用奖励而非惩罚来执行规范（与人类相反），这启发我们思考如何在 AI 系统中引入更接近人类行为的制裁机制，以增强长期合作的稳定性；另一个值得探索的方向是，是否可以通过调整训练数据或提示设计，让推理型模型在社会困境中展现更多合作倾向，例如通过强化学习或模仿人类合作策略。