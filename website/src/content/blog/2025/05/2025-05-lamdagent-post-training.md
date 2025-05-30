---
title: "LaMDAgent: An Autonomous Framework for Post-Training Pipeline Optimization via LLM Agents"
pubDatetime: 2025-05-28T04:30:51+00:00
slug: "2025-05-lamdagent-post-training"
type: "arxiv"
id: "2505.21963"
score: 0.7818024279742551
author: "grok-3-latest"
authors: ["Masafumi Oyamada", "Taro Yano", "Yoichi Ishibashi"]
tags: ["LLM", "Post-Training", "Supervised Fine-Tuning", "Model Merging", "Automation"]
institution: ["NEC Corporation"]
description: "本文提出 LaMDAgent 框架，通过 LLM 智能体自动化构建和优化后训练流程，显著提升模型在多任务和特定技能上的性能，并探索了降低计算成本的策略。"
---

> **Summary:** 本文提出 LaMDAgent 框架，通过 LLM 智能体自动化构建和优化后训练流程，显著提升模型在多任务和特定技能上的性能，并探索了降低计算成本的策略。 

> **Keywords:** LLM, Post-Training, Supervised Fine-Tuning, Model Merging, Automation

**Authors:** Masafumi Oyamada, Taro Yano, Yoichi Ishibashi

**Institution(s):** NEC Corporation


## Problem Background

大型语言模型（LLMs）在各种任务中表现出色，但为了适应特定领域或应用，需要通过后训练技术（如监督微调 SFT、偏好学习、模型合并）进行优化。
目前的后训练流程大多依赖人工设计或仅关注单个组件优化，缺乏端到端的自动化构建，导致高昂的人力成本和对领域专家知识的依赖。
LaMDAgent 旨在通过 LLM 驱动的智能体自动化构建和优化后训练流程，减少人工干预并发现高效策略。

## Method

*   **核心思想:** 提出 LaMDAgent 框架，利用 LLM 智能体自动化构建和优化后训练流程，通过迭代探索不同的模型生成技术、数据集和超参数配置，发现高性能流程。
*   **具体步骤:** 
    *   **动作枚举（Action Enumeration）:** 定义‘对象’（如模型、数据集）和‘动作’（如 SFT、模型合并 TIES），通过组合生成所有可能的动作候选。
    *   **动作选择（Action Selection）:** 利用 LLM 智能体基于历史经验（记忆）和提示模板分步选择动作类型和对象，避免模式崩塌和选择偏差，确保探索的多样性。
    *   **模型评估（Model Evaluation）:** 对生成的模型在目标任务上进行评估，计算单任务或多任务得分作为反馈，指导后续选择。
    *   **记忆更新（Memory Update）:** 根据评估结果更新智能体记忆，总结经验并规划未来探索方向，形成闭环优化。
*   **关键特点:** 将异构的后训练方法统一到框架中，通过智能体的反馈机制动态调整流程，同时探索计算成本优化策略（如数据规模缩放和模型规模缩放）。

## Experiment

*   **实验 1（多技能教学）:** 目标是将多技能（如数学推理、常识推理、阅读理解）融入 Gemma2 2B 模型，LaMDAgent 发现的流程在测试集平均准确率上比最强基线（Fully Fine-Tuned）提升 1.9 个百分点，尤其在数学任务上平均提升 3.7 个百分点，同时保持其他任务竞争力；实验设置涵盖多任务和分布外评估，消融研究表明智能体驱动的动作选择优于随机选择。
*   **实验 2（工具使用技能增强）:** 目标是增强 Gemma2 2B Instruct 的工具使用能力，LaMDAgent 在 AceBench 准确率上提升 9.0 个百分点（从 0.410 到 0.500），MT-Bench 得分几乎不变（0.804 到 0.810），而全数据微调导致性能下降，显示其在分布不匹配场景中的优势。
*   **计算成本优化:** 数据规模缩放被证明有效，Top-1 流程在不同数据规模下保持最佳性能；模型规模缩放存在局限性，小性能差距可能在更大模型上消失。
*   **总体评价:** 实验设置合理，涵盖多场景，数据显著支持 LaMDAgent 的优越性，尤其在复杂任务和分布不匹配情况下效果明显。

## Further Thoughts

LaMDAgent 的智能体驱动方法展示了 LLM 在自动化流程优化中的潜力，是否可以扩展到预训练数据选择或超参数搜索？
动作空间对性能影响显著，是否可以通过引入更多动作类型（如偏好学习、数据生成）或动态扩展动作空间进一步提升性能？
数据规模缩放的有效性提示小规模数据快速迭代后扩展到大规模是否可作为通用计算成本优化策略？
模型规模缩放的局限性是否可以通过设计跨规模迁移策略（如逐步增加模型规模）解决？