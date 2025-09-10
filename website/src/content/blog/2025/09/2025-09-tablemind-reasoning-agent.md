---
title: "TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning"
pubDatetime: 2025-09-08T02:00:31+00:00
slug: "2025-09-tablemind-reasoning-agent"
type: "arxiv"
id: "2509.06278"
score: 0.7408230308728952
author: "grok-3-latest"
authors: ["Chuang Jiang", "Mingyue Cheng", "Xiaoyu Tao", "Qingyang Mao", "Jie Ouyang", "Qi Liu"]
tags: ["LLM", "Table Reasoning", "Tool Integration", "Reinforcement Learning", "Autonomous Agent"]
institution: ["State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China"]
description: "TableMind 提出了一种自主的表格推理代理，通过‘计划-行动-反思’循环和两阶段训练范式，显著提升了结构化数据处理任务的性能和适应性。"
---

> **Summary:** TableMind 提出了一种自主的表格推理代理，通过‘计划-行动-反思’循环和两阶段训练范式，显著提升了结构化数据处理任务的性能和适应性。 

> **Keywords:** LLM, Table Reasoning, Tool Integration, Reinforcement Learning, Autonomous Agent

**Authors:** Chuang Jiang, Mingyue Cheng, Xiaoyu Tao, Qingyang Mao, Jie Ouyang, Qi Liu

**Institution(s):** State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China


## Problem Background

表格推理在金融、医疗和科研等领域处理结构化数据至关重要，但大型语言模型（LLMs）在复杂数值计算和精细操作上表现不佳。
现有工具增强推理方法依赖固定模式和监督模仿，缺乏自主适应能力，导致在多样化表格任务和查询类型上的灵活性不足。

## Method

*   **核心框架：** TableMind 是一个基于 LLM 的表格推理代理，通过自主的多轮工具调用和‘计划-行动-反思’循环实现灵活问题解决。
*   **具体实现：**
    *   **计划阶段：** 模型分析查询和当前状态，分解复杂问题为可执行步骤，制定下一步策略。
    *   **行动阶段：** 生成 Python 代码用于数据处理或数值计算，并在安全沙箱环境中执行。
    *   **反思阶段：** 根据执行结果（成功输出或错误信息）评估进展，动态调整后续计划，直至得出最终答案。
*   **训练策略：**
    *   **监督微调（SFT）：** 使用高质量推理轨迹数据（如通过专家模型蒸馏生成），训练模型掌握工具调用语法和基本推理模式，作为初始策略。
    *   **强化微调（RFT）：** 设计多目标奖励函数（包括格式正确性、答案准确性和工具交互效率），并提出 Rank-Aware Policy Optimization (RAPO) 算法，通过对高质量轨迹的动态加权优化策略探索效率。
*   **关键创新：** 从固定流程转向动态代理，模拟人类分析师的灵活性和错误纠正能力，同时通过两阶段训练平衡基础能力和策略优化。

## Experiment

*   **性能表现：** 在 WikiTQ、TabMWP 和 TabFact 三个基准数据集上，TableMind 显著优于基线模型（如 Tab-CoT, PoTable, Chain-of-Table），分别提升了 2.61%、3.22% 和 3.28%。
*   **实验设置：** 涵盖开放域问答、数学推理和事实验证任务，数据集选择合理，测试集用于评估确保公平性。
*   **消融研究：** 验证了 SFT 和 RFT 阶段的必要性，移除任一阶段均导致性能下降；RAPO 算法和多目标奖励设计（如工具交互奖励）对提升策略效率和最终准确性至关重要。
*   **工具使用分析：** 工具调用比例从初始波动到接近 100%，代码执行成功率从 60% 提升至近 90%，表明模型逐步优化了工具交互策略。
*   **总结：** 方法提升明显，实验设计全面合理，充分展示了 TableMind 在准确性和适应性上的优势。

## Further Thoughts

TableMind 的‘计划-行动-反思’循环机制启发我们在其他复杂任务（如多模态数据处理或实时决策）中引入动态适应和自我纠正能力；两阶段训练范式（SFT+RFT）可推广至其他需要工具交互的 LLM 应用，如自动化代码生成；RAPO 算法的动态加权策略可能适用于更广泛的强化学习场景，提升探索效率。