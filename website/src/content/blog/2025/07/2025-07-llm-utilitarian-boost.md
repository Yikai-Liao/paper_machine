---
title: "Many LLMs Are More Utilitarian Than One"
pubDatetime: 2025-07-01T14:46:16+00:00
slug: "2025-07-llm-utilitarian-boost"
type: "arxiv"
id: "2507.00814"
score: 0.542339290898205
author: "grok-3-latest"
authors: ["Anita Keshmirian", "Razan Baltaji", "Babak Hemmatian", "Hadi Asghari", "Lav R. Varshney"]
tags: ["LLM", "Multi-Agent Systems", "Moral Reasoning", "Utilitarian Boost", "Group Dynamics"]
institution: ["Forward College", "University of Illinois at Urbana-Champaign", "University of Nebraska, Lincoln", "Technische Universität Berlin"]
description: "本文首次系统揭示了大型语言模型在多代理系统中表现出集体道德决策的功利主义提升现象，并发现其机制与人类差异显著，为 AI 对齐和多代理系统设计提供了关键洞见。"
---

> **Summary:** 本文首次系统揭示了大型语言模型在多代理系统中表现出集体道德决策的功利主义提升现象，并发现其机制与人类差异显著，为 AI 对齐和多代理系统设计提供了关键洞见。 

> **Keywords:** LLM, Multi-Agent Systems, Moral Reasoning, Utilitarian Boost, Group Dynamics

**Authors:** Anita Keshmirian, Razan Baltaji, Babak Hemmatian, Hadi Asghari, Lav R. Varshney

**Institution(s):** Forward College, University of Illinois at Urbana-Champaign, University of Nebraska, Lincoln, Technische Universität Berlin


## Problem Background

随着大型语言模型（LLMs）被广泛整合到多代理系统（Multi-Agent Systems, MAS）中，用于处理医疗、法律等高风险领域的复杂任务，理解其集体道德推理行为变得至关重要。
论文关注一个关键问题：LLM 在群体协作时是否会像人类群体一样，表现出功利主义提升（Utilitarian Boost），即更倾向于为大多数人利益而牺牲少数人？
此外，这种现象背后的机制是否与人类一致？如果不同，可能对 AI 安全和伦理对齐带来哪些风险？
研究指出，忽视群体层面的道德动态可能导致无法预测或防止伦理问题，尤其是在高风险决策中。

## Method

*   **实验设计核心:** 对比单个 LLM（Solo 条件）和多代理 LLM 群体（Group 条件）在道德困境中的决策行为，探索集体推理是否导致功利主义倾向的提升。
*   **具体步骤:**
    *   选取六种主流 LLM（包括 Llama3.3-70B、QwQ、Qwen3-32B、Gemma3-27B、Qwen2.5-32B 和 GPT4.1）进行测试。
    *   使用经典道德困境场景（如个人 vs. 非个人困境、行动 vs. 不行动困境）作为测试基准，覆盖多种道德维度。
    *   在 Solo 条件下，模型独立对道德困境进行推理并给出道德可接受性评分（1-7 分）。
    *   在 Group 条件下，模型以二或三人小组进行多轮讨论（共 6 轮），逐步达成共识并更新评分，同时在讨论后进行私人反思。
    *   引入心理学测量工具，如 Oxford Utilitarianism Scale（区分公正受益和工具性伤害）、CNI 问卷（评估结果敏感性、规范敏感性和不行动偏好），以量化功利主义倾向及其机制。
    *   使用混合效应回归模型分析 Solo 和 Group 条件下的评分差异，并探讨不同模型在机制上的异同。
*   **创新点:** 结合社会心理学实验设计和计算分析方法，系统研究 LLM 集体道德推理的表象和内在驱动因素，弥补了以往研究多聚焦于单个模型的不足。

## Experiment

*   **有效性:** 实验结果表明，所有六种 LLM 在 Group 条件下均表现出显著的功利主义提升，尤其在个人道德困境中（直接伤害一人以拯救更多人），评分显著高于 Solo 条件（β = 0.31, p < 0.0001）。
*   **机制差异:** 与人类群体不同（人类主要因对结果的敏感性提升而更功利），LLM 的功利主义提升机制因模型而异：如 QwQ 更关注结果最大化，GPT4.1 在高收益时更愿意违反规范，Llama3.3 对规范和行动/不行动区分不敏感。
*   **设置合理性:** 实验覆盖多种道德困境维度，测试了多个主流模型，并通过重复试验（每试验重复三次）和可靠性检查（人类评分验证一致性）增强了结果的可信度；但局限性在于未明确识别导致功利主义提升的具体语言触发因素，也未探讨群体初始道德立场多样性对结果的影响。
*   **结论:** 实验全面揭示了 LLM 集体道德推理的功利主义倾向，但其机制与人类差异显著，提示需针对性设计 AI 对齐策略。

## Further Thoughts

论文揭示了 LLM 集体道德推理行为虽表面上模仿人类功利主义倾向，但内在机制显著不同，这启发我们：AI 伦理对齐不能简单套用人类道德模型，而应针对 LLM 独特机制设计干预措施。例如，可以探索通过引入对抗性或异议代理来打破群体中的功利主义强化循环，防止潜在伦理风险；此外，不同模型的机制差异可能与训练数据或微调策略有关，未来可研究如何通过调整训练或推理参数，控制集体道德行为，确保其与人类价值观更一致。