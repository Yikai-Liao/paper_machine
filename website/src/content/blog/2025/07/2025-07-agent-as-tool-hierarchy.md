---
title: "Agent-as-Tool: A Study on the Hierarchical Decision Making with Reinforcement Learning"
pubDatetime: 2025-07-02T08:49:43+00:00
slug: "2025-07-agent-as-tool-hierarchy"
type: "arxiv"
id: "2507.01489"
score: 0.7459324316576464
author: "grok-3-latest"
authors: ["Yanfei Zhang"]
tags: ["LLM", "Hierarchical Framework", "Reinforcement Learning", "Tool Calling", "Reasoning"]
institution: ["Independent Researcher"]
description: "本文提出 **Agent-as-Tool** 分层框架，通过分离推理和工具调用职责并结合强化学习优化，在多跳推理任务中显著提升性能，尤其在 Bamboogle 数据集上超越现有最佳方法。"
---

> **Summary:** 本文提出 **Agent-as-Tool** 分层框架，通过分离推理和工具调用职责并结合强化学习优化，在多跳推理任务中显著提升性能，尤其在 Bamboogle 数据集上超越现有最佳方法。 

> **Keywords:** LLM, Hierarchical Framework, Reinforcement Learning, Tool Calling, Reasoning

**Authors:** Yanfei Zhang

**Institution(s):** Independent Researcher


## Problem Background

大型语言模型（LLMs）在复杂任务中常通过外部工具（如搜索引擎、计算器）增强能力，但现有强化学习（RL）增强的代理框架将工具调用和推理过程紧密耦合，导致训练难度增加和推理质量下降。
具体问题包括：模型需同时学习工具选择、输入构建和推理，增加了噪声；工具返回的原始结果往往包含冗余信息，影响推理准确性。
论文的出发点是通过分层设计解耦这两个过程，提升多跳推理任务的表现。

## Method

*   **核心思想:** 提出 **Agent-as-Tool** 框架，通过分层设计将推理和工具调用职责分离，降低模型负担，提升推理质量。
*   **框架组成:** 
    *   **Planner:** 负责高层次推理和任务分解，使用自然语言进行思考（以 `<think>...</think>` 标记）和工具调用指令（以 `<tool calling>...</tool calling>` 标记），基于工具返回的结构化观察进行下一步决策。
    *   **Toolcaller:** 负责与外部工具交互（如调用搜索引擎），并将工具返回的原始结果处理为结构化观察（以 `<obs>...</obs>` 标记）反馈给 Planner，当前实现基于 CAMEL 风格的聊天代理和 GPT-4o-mini 模型。
*   **训练方法:** 
    *   使用 Generalized Reinforcement Policy Optimization (GRPO) 对 Planner 进行强化学习微调，目标是优化推理决策。
    *   引入观察掩码（observation masking），在训练时用特殊标记 `<fimpad>` 替换工具返回的观察内容，防止奖励泄露，确保信用分配的准确性。
    *   奖励函数设计结合答案正确性和格式约束，正确且格式符合要求的输出获得高奖励，否则给予惩罚。
*   **训练规模:** 仅使用 180 个样本（来自 HotpotQA 和 2WikiMultiHopQA 数据集）进行微调，体现框架的高效性。

## Experiment

*   **有效性:** 在多个多跳问答数据集上，**Agent-as-Tool** 表现出色，尤其在 Bamboogle 数据集上达到 EM 63.2% 和 CEM 75.2%，分别比最强基线 Search-R1 提升 4.8% 和 3.2%。
*   **强化微调效果:** 微调后模型性能平均提升 EM 2.5% 和 CEM 2.3%，表明 GRPO 优化对 Planner 的推理能力有显著帮助。
*   **对比分析:** 相较于 CAMEL Agent 和 Direct IO 等基线，**Agent-as-Tool** 在大多数指标上表现优异；相较于 Search-R1，虽然在 HotpotQA 等数据集的部分指标上略逊，但整体性能接近且在 Bamboogle 上显著领先。
*   **定性优势:** 分层设计使推理过程更清晰，Planner 基于结构化观察进行推理，避免了直接处理工具返回的模糊、冗余信息带来的干扰。
*   **实验设置合理性:** 实验覆盖多个数据集（HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle）和多种基线模型，设置较为全面；但训练规模较小（仅 180 样本），可能未完全挖掘模型潜力，损失在前 30 步不稳定也反映了数据量不足的影响。

## Further Thoughts

分层设计的模块化思想非常具有启发性，分离推理和工具调用职责的框架可以推广到其他复杂任务中，例如多工具协同场景或动态任务分配；此外，论文提到的将 Planner 升级为工具编排者（Tool Orchestrator）的设想启发了我思考如何通过强化学习优化工具选择策略，是否可以设计更复杂的奖励函数以适应不同任务需求；另一个方向是探索是否可以通过增加训练数据多样性或引入多模型协作，进一步提升分层框架的泛化能力。