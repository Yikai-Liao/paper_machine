---
title: "LLM-Augmented Chemical Synthesis and Design Decision Programs"
pubDatetime: 2025-05-11T15:43:00+00:00
slug: "2025-05-llm-retrosynthesis-planner"
type: "arxiv"
id: "2505.07027"
score: 0.49789158105104353
author: "grok-3-latest"
authors: ["Haorui Wang", "Jeff Guo", "Lingkai Kong", "Rampi Ramprasad", "Philippe Schwaller", "Yuanqi Du", "Chao Zhang"]
tags: ["LLM", "Retrosynthesis", "Molecular Design", "Evolutionary Search", "Planning"]
institution: ["Georgia Tech", "École Polytechnique Fédérale de Lausanne (EPFL)", "National Centre of Competence in Research (NCCR) Catalysis", "Harvard University", "Cornell University"]
description: "本文提出 LLM-Syn-Planner 框架，通过直接生成并优化多步逆合成路径，显著提升大型语言模型在逆合成规划中的表现，并成功扩展到可合成分子设计问题。"
---

> **Summary:** 本文提出 LLM-Syn-Planner 框架，通过直接生成并优化多步逆合成路径，显著提升大型语言模型在逆合成规划中的表现，并成功扩展到可合成分子设计问题。 

> **Keywords:** LLM, Retrosynthesis, Molecular Design, Evolutionary Search, Planning

**Authors:** Haorui Wang, Jeff Guo, Lingkai Kong, Rampi Ramprasad, Philippe Schwaller, Yuanqi Du, Chao Zhang

**Institution(s):** Georgia Tech, École Polytechnique Fédérale de Lausanne (EPFL), National Centre of Competence in Research (NCCR) Catalysis, Harvard University, Cornell University


## Problem Background

逆合成（Retrosynthesis）规划是有机化学和药物开发中的核心问题，旨在将目标分子分解为可购买的前体分子，但由于合成路径的搜索空间随反应步骤呈指数增长，传统单步预测模型和搜索算法难以高效解决多步规划问题。
近年来，大型语言模型（LLMs）展现出化学知识和复杂决策潜力，但其在高度约束的逆合成任务中的应用仍未被充分探索，作者希望利用 LLMs 的生成和规划能力，解决逆合成规划及更广泛的可合成分子设计挑战。

## Method

* **LLM 作为单步预测模型：** 将 LLM 集成到传统搜索算法（如 Monte Carlo Tree Search 或 Retro*）中，用于预测每一步的反应模板或反应物。
  * 由于 LLM 难以直接输出反应概率，作者通过‘自我一致性频率’方法为预测分配伪概率，即采样多次独立反应并计算频率作为概率估计。
  * 结合参考反应数据库，通过检索避免虚构反应模板，确保化学可行性。
* **LLM-Syn-Planner（进化搜索算法）：** 这是核心创新，直接利用 LLM 生成完整的多步合成路径，并通过进化算法优化。
  * **初始化（Initialization）：** 基于目标分子，通过分子相似性（Morgan 指纹和 Tanimoto 相似性）检索参考路径，提供给 LLM 生成初始路径池。
  * **评估（Evaluation）：** 设计三级反馈机制（分子级别：验证分子有效性和可购买性；反应级别：匹配数据库验证反应可行性；路径级别：检查路径连通性），评估每一步质量。
  * **选择（Selection）：** 引入部分奖励机制（基于 SC 分数评估无效步骤的分子集），选择适应度最高的路径进入下一轮。
  * **变异（Mutation）：** 利用 LLM 修改无效步骤，结合评估反馈和参考路径，生成新的候选路径，重复迭代直至找到解或达到预算。
  * 采用线性格式表示合成路径，降低树形结构的复杂性，并引入‘Rational’推理组件，鼓励 LLM 在决策前思考，提升一致性。
* **扩展到可合成分子设计：** 提出 LLM-Syn-Designer，结合分子优化工具（如 MolLEO），在优化分子属性的同时通过 SC 分数过滤确保可合成性，形成端到端设计框架。

## Experiment

* **逆合成规划：** 在 USPTO 和 Pistachio 数据集上，LLM 作为单步预测模型时，解决率远低于传统模型（如 Graph2Edits, LocalRetro），尤其在困难数据集（如 Pistachio Hard）上接近于零；而 LLM-Syn-Planner 表现突出，解决率接近甚至超过传统方法（如 USPTO Easy 上达 93%-100%），表明直接生成并优化完整路径更适合 LLMs。
* **可合成分子设计：** LLM-Syn-Designer 在优化分子属性（如 JNK3, GSK3B）时，相比传统方法（如 Graph-GA, REINVENT），在更少的 oracle 调用下达到类似或更高的适应度分数，同时保证了可合成性。
* **消融研究：** 线性路径格式显著优于树形格式（成功率提升至 91%）；分子 RAG 即使提供随机参考路径也能提升性能（成功率从 51% 提升至 84%）；部分奖励机制对长距离决策至关重要（成功率从 63.5% 提升至 91%）；提示设计中的解释性部分进一步提升表现。
* **实验设置合理性：** 数据集涵盖不同难度，基线模型包括多种传统方法，实验限制（如 60 分钟搜索时间）确保公平性，数据表明 LLM-Syn-Planner 的提升显著，尤其在完整路径规划上的突破。

## Further Thoughts

线性路径格式显著降低了合成路径复杂性，启发在其他树形搜索问题中尝试类似简化表示；部分奖励机制对 LLM 长距离决策至关重要，可在其他序列决策任务中设计中间反馈；检索增强生成（RAG）即使提供随机参考路径也能提升性能，表明其价值在于上下文范例，可在代码生成或知识推理中探索；进化搜索与 LLM 结合展现强大生成与优化能力，未来可应用于材料约束合成规划或其他多目标优化问题。