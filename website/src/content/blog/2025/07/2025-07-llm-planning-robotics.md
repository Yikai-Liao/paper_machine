---
title: "Can LLM-Reasoning Models Replace Classical Planning? A Benchmark Study"
pubDatetime: 2025-07-31T14:25:54+00:00
slug: "2025-07-llm-planning-robotics"
type: "arxiv"
id: "2507.23589"
score: 0.5928943964002763
author: "grok-3-latest"
authors: ["Kai Goebel", "Patrik Zips"]
tags: ["LLM", "Planning", "Robotics", "Hybrid Planning", "Reasoning"]
institution: ["AIT Austrian Institute of Technology GmbH"]
description: "本文通过基准测试系统性评估了大型语言模型（LLMs）在机器人任务规划中的性能，揭示其在简单任务中的潜力与复杂场景中的局限，倡导混合规划方法以提升可靠性和可扩展性。"
---

> **Summary:** 本文通过基准测试系统性评估了大型语言模型（LLMs）在机器人任务规划中的性能，揭示其在简单任务中的潜力与复杂场景中的局限，倡导混合规划方法以提升可靠性和可扩展性。 

> **Keywords:** LLM, Planning, Robotics, Hybrid Planning, Reasoning

**Authors:** Kai Goebel, Patrik Zips

**Institution(s):** AIT Austrian Institute of Technology GmbH


## Problem Background

随着大型语言模型（LLMs）在常识推理和多步骤输出生成方面的进步，其在机器人任务规划中的潜力受到关注。然而，LLMs 生成的计划在结构化和可执行性方面存在不确定性，尤其是在需要精确资源管理、状态跟踪和严格约束的复杂场景中，执行失败可能导致安全风险或任务失败。本文旨在探讨 LLMs 是否能够替代经典规划方法（如 Fast Downward），并识别其在机器人规划中的优势与局限。

## Method

* **核心设计：** 本研究通过基准测试方法，系统性比较了九种当前最先进的 LLMs（如 GPT-o1、Claude Sonnet 3.7 Thinking）与经典规划器 Fast Downward 在机器人任务规划中的性能。
* **测试领域：** 选择了五个 PDDL（Planning Domain Definition Language）领域（barman, blocks, elevator, satellite, tidybot），这些领域涵盖了从简单动作排序到复杂资源管理和并发性的不同挑战。
* **提示设计：** LLMs 被直接提示使用 PDDL 域和问题文件生成计划，提示要求模型输出高层次推理概述和具体的动作序列，并以 JSON 格式结构化输出（包括动作名称、参数、选择理由和验证声明）。
* **评估指标：** 采用多维度指标评估性能，包括成功率（Success Rate，计划是否达成目标）、计划长度（Plan Length，动作数量）、可执行动作数（Executed Actions，计划中可实际执行的动作数）、执行保真度（Execution Fidelity，可执行动作与计划动作的比率）以及规划时间（Planning Time，生成计划所需时间）。
* **对比基准：** Fast Downward 使用 `seq-sat-lama-2011` 启发式配置，结合地标和 FF 启发式，通过迭代搜索生成高质量计划，规划时间限制为 600 秒，作为经典规划的可靠基准。

## Experiment

* **成功率：** LLMs 在简单任务（如 blocks 和 elevator 领域）中表现较好，GPT-o1 和 Claude Sonnet 3.7 Thinking 的平均成功率达到 63.4%，但在复杂领域（如 barman 和 tidybot）中显著低于 Fast Downward（平均成功率 97.85%）。
* **执行保真度：** GPT-o1 的执行保真度最高（73.4%），表明其生成的计划中有较高比例的动作可执行，但仍远低于 Fast Downward 的 100%。其他模型如 Llama 405B Instruct 表现极差（仅 13.9%）。
* **规划时间：** LLMs 的规划时间差异较大，Claude Sonnet 3.5 最快（14.22 秒），而 Llama DeepSeek R1 最慢（160.15 秒），且时间长短与成功率无直接相关性，相比之下 Fast Downward 在 600 秒内始终生成高质量计划。
* **实验设置合理性：** 实验覆盖了多种复杂度的领域和多维度指标，设置较为全面，但未深入分析 LLMs 内部推理过程，也未探讨微调或提示优化对性能的影响。
* **结论：** LLMs 在简单规划任务中展现潜力，但在复杂场景中仍不敌经典规划器，特别是在资源协调和约束遵守方面存在明显不足。

## Further Thoughts

论文提出的混合规划（Hybrid Planning）方法启发了我，是否可以设计一个分层框架，让 LLMs 负责高层次策略生成（如目标分解和初步计划），而将低层次动作序列的验证和优化交给经典规划器，以兼顾灵活性和可靠性？此外，迭代计划精炼（Iterative Plan Refinement）的思路也值得探索，是否可以通过引入实时环境反馈机制，让 LLMs 在执行过程中动态调整计划，从而更贴近人类在长程任务中的适应性行为？