---
title: "Meta-Policy Reflexion: Reusable Reflective Memory and Rule Admissibility for Resource-Efficient LLM Agent"
pubDatetime: 2025-09-04T08:18:39+00:00
slug: "2025-09-meta-policy-reflexion"
type: "arxiv"
id: "2509.03990"
score: 0.6696516788268515
author: "grok-3-latest"
authors: ["Chunlong Wu", "Zhibo Qu"]
tags: ["LLM", "Memory Mechanism", "Reflection", "Agent Behavior", "Rule Admissibility"]
institution: ["Tongji University"]
description: "本文提出 Meta-Policy Reflexion (MPR) 框架，通过外部 Meta-Policy Memory 存储结构化反思规则，并在推理时结合软指导和硬约束提升 LLM 智能体的准确性和安全性，实现资源高效的自改进。"
---

> **Summary:** 本文提出 Meta-Policy Reflexion (MPR) 框架，通过外部 Meta-Policy Memory 存储结构化反思规则，并在推理时结合软指导和硬约束提升 LLM 智能体的准确性和安全性，实现资源高效的自改进。 

> **Keywords:** LLM, Memory Mechanism, Reflection, Agent Behavior, Rule Admissibility

**Authors:** Chunlong Wu, Zhibo Qu

**Institution(s):** Tongji University


## Problem Background

大型语言模型（LLM）作为智能体在单任务上表现优异，但存在重复失败、低效探索和跨任务适应性差的问题；现有的反思方法（如 Reflexion）生成的临时性、任务特定痕迹无法复用，而强化学习（RL）方法虽能生成可复用策略，但计算成本高；本文旨在设计一种轻量级框架，将反思提炼为可复用的结构化知识，并在不调整模型参数的情况下提升行为的安全性和泛化能力。

## Method

* **核心框架：Meta-Policy Reflexion (MPR)**：提出一个外部的 Meta-Policy Memory (MPM)，以谓词形式存储从失败轨迹中提炼的结构化规则，用于指导智能体行为。
* **基础策略**：将任务建模为马尔可夫决策过程（MDP），LLM 作为固定策略生成器，根据当前状态生成动作，参数保持冻结。
* **软指导（Soft Guidance）**：在推理时，检索与当前状态相关的 MPM 规则，注入到 LLM 提示中，引导其生成符合历史经验的动作；此过程仅在提示层面操作，不修改模型内部 logits 或概率分布。
* **硬可接受性检查（Hard Admissibility Check, HAC）**：在动作生成后，通过环境或用户定义的约束集验证动作合法性，若不满足则重新采样或选择默认安全动作，确保行为符合领域约束。
* **无训练自改进**：任务结束后，通过 LLM 反思功能分析失败轨迹，提取新的谓词规则更新 MPM，逐步积累知识，无需梯度更新或模型微调。
* **算法流程**：分为训练阶段（从失败中更新 MPM）和推理阶段（冻结 MPM 进行软指导和硬检查），实现知识获取与安全执行的解耦。

## Experiment

* **实验设置**：在 AlfWorld 文本交互环境上进行测试，使用 Qwen3-32b 作为基础 LLM，对比 Reflexion 基线；实验分为训练阶段（更新 MPM）和推理阶段（冻结 MPM），采用固定随机种子确保一致性。
* **训练集表现**：在 60 个任务的训练集上，MPR 从第 1 轮准确率 83.9% 快速提升至第 3 轮的 100%，显著优于 Reflexion（从 70.0% 到第 5 轮的 88.3%），显示规则提取的高效性。
* **测试集泛化**：在 74 个任务的测试集上，MPR（训练 5 轮后测试 1 次）准确率为 87.8%，略高于 Reflexion（测试集反思 6 轮）的 86.9%；加入 HAC 后准确率提升至 91.4%，表明硬约束增强了行为可靠性。
* **分析与局限**：实验设置合理，涵盖训练测试分离和多轮对比，但 AlfWorld 任务结构规律性强，可能导致 MPR 快速收敛；在更异构领域可能需更多轮次或复杂规则表示，适用性待验证。

## Further Thoughts

MPR 的外部记忆层（MPM）概念启发了我对多智能体协作的思考，规则共享可能实现群体学习；软硬结合机制在高风险领域（如医疗）有潜力，未来可探索动态约束阈值或人类验证；此外，MPM 的规则积累是否能通过置信度加权或抽象化优化，形成更高效的终身学习体系，值得进一步研究。