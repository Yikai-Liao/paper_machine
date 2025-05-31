---
title: "LaMDAgent: An Autonomous Framework for Post-Training Pipeline Optimization via LLM Agents"
pubDatetime: 2025-05-28T04:30:51+00:00
slug: "2025-05-lamdagent-post-training"
type: "arxiv"
id: "2505.21963"
score: 0.7818024279742551
author: "grok-3-latest"
authors: ["Masafumi Oyamada", "Taro Yano", "Yoichi Ishibashi"]
tags: ["LLM", "Post-Training", "Model Merging", "Supervised Fine-Tuning", "Autonomous Agents"]
institution: ["NEC Corporation"]
description: "本文提出 LaMDAgent，一个基于 LLM 代理的自主框架，自动化构建和优化后训练管道，显著提升模型性能并减少人工干预依赖。"
---

> **Summary:** 本文提出 LaMDAgent，一个基于 LLM 代理的自主框架，自动化构建和优化后训练管道，显著提升模型性能并减少人工干预依赖。 

> **Keywords:** LLM, Post-Training, Model Merging, Supervised Fine-Tuning, Autonomous Agents

**Authors:** Masafumi Oyamada, Taro Yano, Yoichi Ishibashi

**Institution(s):** NEC Corporation


## Problem Background

大型语言模型（LLMs）在后训练（Post-Training）阶段需要针对特定领域或任务进行适配，常见方法包括监督微调（Supervised Fine-Tuning, SFT）、偏好学习和模型合并（Model Merging），但目前管道设计主要依赖人工经验，缺乏自动化和系统化方法，导致效率低下且对领域专家依赖度高。
论文旨在解决如何自动化构建并优化后训练管道这一关键问题，以减少人工干预并提升模型在目标任务上的性能。

## Method

*   **核心思想:** 提出 LaMDAgent（Language Model Developing Agent），一个基于 LLM 代理的自主框架，通过迭代探索和任务反馈，自动化构建和优化后训练管道。
*   **具体实现步骤:**
    *   **动作枚举（Action Enumeration）**：定义‘对象’（如模型、数据集）和‘动作’（如 SFT、模型合并），通过枚举动作类型和对象的组合生成所有可能的改进策略。
    *   **动作选择（Action Selection）**：利用 LLM 代理分步选择动作类型和对象，通过设计提示词鼓励探索性行为，避免模式崩塌（Mode Collapse）或选择偏差，同时提高解析准确性。
    *   **模型评估（Model Evaluation）**：对生成的模型在目标任务上进行评估，计算单任务或多任务得分（通过加权聚合不同任务的指标），作为反馈提供给代理。
    *   **记忆更新（Memory Update）**：基于评估结果更新代理的记忆，记录过去尝试的经验和未来探索方向，形成动态调整策略的闭环。
*   **关键创新:** 将异构的后训练方法统一到单一框架中，通过代理的自主探索和反馈机制动态优化管道，显著减少对专家知识的依赖，同时支持多种任务场景。

## Experiment

*   **实验 1（多技能教学）**：目标是将多个技能（如数学推理、常识推理、阅读理解）融入基础模型（Gemma2 2B），LaMDAgent 发现的管道在测试集上的平均准确率比最强基线（Fully Fine-Tuned）高出 1.9 个百分点，尤其在数学相关任务上提升了 3.7 个百分点，同时在其他任务上保持竞争力；消融研究表明基于 LLM 的动作选择比随机选择更有效，动作空间设计对性能影响显著。
*   **实验 2（工具使用技能增强）**：目标是增强指令微调模型（Gemma2 2B Instruct）的工具使用能力，LaMDAgent 将 AceBench 准确率提升了 9.0 个百分点（从 0.410 到 0.500），而 MT-Bench 得分几乎不变（0.804 到 0.810），相比之下简单微调方法导致性能下降，表明 LaMDAgent 在分布不匹配的情况下更有效。
*   **计算成本降低策略**：数据规模扩展（Data Size Scaling）被证明有效，优化的管道在小数据量上的优势可扩展到大数据量；而模型规模扩展（Model Size Scaling）存在挑战，小规模模型上的性能差距在扩展到大规模模型时可能消失。
*   **实验设置评价**：实验涵盖多任务场景和现实应用（如工具使用），包括分布内和分布外评估，设置全面合理，数据支持了方法的显著性提升。

## Further Thoughts

LaMDAgent 的代理驱动优化思路启发我们将 LLM 代理的自主探索能力应用于其他复杂优化问题，如超参数搜索或架构设计；其反馈驱动的记忆更新机制类似于强化学习中的经验回放，可能为自适应学习系统提供新思路；此外，数据规模扩展的有效性提示可以在资源受限的情况下先在小规模上快速迭代优化策略，再扩展到大规模资源，具有实际应用价值。