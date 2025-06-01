---
title: "Fortune: Formula-Driven Reinforcement Learning for Symbolic Table Reasoning in Language Models"
pubDatetime: 2025-05-29T17:13:40+00:00
slug: "2025-05-formula-tuning-rl"
type: "arxiv"
id: "2505.23667"
score: 0.7415707434212159
author: "grok-3-latest"
authors: ["Lang Cao", "Jingxian Xu", "Hanbing Liu", "Jinyu Wang", "Mengyu Zhou", "Haoyu Dong", "Shi Han", "Dongmei Zhang"]
tags: ["LLM", "Symbolic Reasoning", "Reinforcement Learning", "Table Understanding", "Formula Generation"]
institution: ["University of Illinois Urbana-Champaign", "Nankai University", "Tsinghua University", "Shandong University", "Microsoft Research"]
description: "本文提出 *Formula Tuning* (Fortune)，一种基于强化学习的框架，通过生成可执行电子表格公式显著提升语言模型在表格数据上的符号推理能力，尤其在复杂多步任务中表现优异。"
---

> **Summary:** 本文提出 *Formula Tuning* (Fortune)，一种基于强化学习的框架，通过生成可执行电子表格公式显著提升语言模型在表格数据上的符号推理能力，尤其在复杂多步任务中表现优异。 

> **Keywords:** LLM, Symbolic Reasoning, Reinforcement Learning, Table Understanding, Formula Generation

**Authors:** Lang Cao, Jingxian Xu, Hanbing Liu, Jinyu Wang, Mengyu Zhou, Haoyu Dong, Shi Han, Dongmei Zhang

**Institution(s):** University of Illinois Urbana-Champaign, Nankai University, Tsinghua University, Shandong University, Microsoft Research


## Problem Background

大型语言模型（LLMs）在自然语言处理任务中表现出色，但在表格数据上的数值和符号推理能力不足，尤其是在复杂多步推理场景中。
表格作为数据组织和分析的基础结构，对智能系统至关重要，但现有模型常依赖模式记忆而非规则学习，导致数学计算错误频发；符号方法虽提高精度，但因推理能力有限难以泛化。
电子表格公式是一种灵活且强大的符号操作工具，但当前模型生成准确公式的能力不足，且相关数据集较为简单，难以满足复杂推理需求。

## Method

*   **核心思想:** 提出 *Formula Tuning* (Fortune)，一种基于强化学习（RL）的框架，训练语言模型生成可执行的电子表格公式，用于表格数据的符号推理和问题解答。
*   **具体实现:**
    *   使用答案正确性作为奖励信号，减少对监督公式标注的依赖，通过强化学习（如 Proximal Policy Optimization, PPO）引导模型探索并生成正确公式。
    *   模型首先生成推理轨迹（chain of thought），随后生成公式，公式通过确定性执行引擎计算最终答案，依据答案正确性分配奖励（正确为1，错误但可执行为0.2，不可执行为0）。
    *   采用监督微调（SFT）作为冷启动策略，为强化学习提供初始知识，提升训练稳定性和性能上限。
    *   提出 *Fortune++*，在推理时联合文本推理和符号推理（生成5个文本输出和5个公式输出），通过自一致性策略（self-consistency）投票选择最终答案。
*   **理论依据:** 符号推理优于文本推理（通过确定性执行避免数值错误），强化学习优于监督微调（能探索超出教师策略的高奖励解）。
*   **关键优势:** 不依赖大量标注数据，允许模型探索多样化解决方案，提升复杂推理任务的准确性和泛化能力。

## Experiment

*   **有效性:** *Formula Tuning* 和 *Fortune++* 在七个表格推理基准数据集上显著提升性能，尤其在多步数值和符号推理任务中表现突出；例如，*Fortune++* 在 FinQA 上达到 80.47% 准确率，在 AIT-QA 上达到 93.20%，相比零样本设置提升明显。
*   **对比分析:** 强化学习（RL）相较监督微调（SFT）进一步提升公式生成能力，尤其在分布外（OOD）数据集（如 AIT-QA, TableBench）上泛化性更强；冷启动策略（SFT+RL）使开源模型（如 Qwen2.5-Coder 7B）整体准确率提升至 68.4%，接近甚至超越部分闭源模型（如 GPT-4o-mini）。
*   **互补性:** *Fortune++* 联合文本和符号推理在所有基准上表现最佳，证明两种推理方式在不同任务场景（如简单查询 vs. 复杂计算）中的互补优势。
*   **实验设置:** 实验覆盖多种模型（开源和闭源）、数据集（分布内和分布外），并与多种基线方法（如 TAPEX, TabAF）对比，设置较为全面合理；但数据集以结构化表格为主，未充分覆盖真实世界中的噪声或多模态数据。

## Further Thoughts

电子表格公式作为符号推理媒介的灵活性和图灵完备性为复杂表格推理提供了新思路，是否可扩展至其他符号系统或图形化推理领域？
强化学习直接优化任务奖励的思想是否适用于其他结构化输出任务（如代码生成、逻辑推理）？
联合文本和符号推理的策略是否可以通过自适应机制根据任务复杂度动态选择推理路径？
奖励设计目前仅基于答案正确性，是否可引入细粒度奖励（如公式简洁性、计算效率）以优化生成质量？