---
title: "Large Language Models for Planning: A Comprehensive and Systematic Survey"
pubDatetime: 2025-05-26T08:44:53+00:00
slug: "2025-05-llm-planning-survey"
type: "arxiv"
id: "2505.19683"
score: 0.5533020121381825
author: "grok-3-latest"
authors: ["Pengfei Cao", "Tianyi Men", "Wencan Liu", "Jingwen Zhang", "Xuzhao Li", "Xixun Lin", "Dianbo Sui", "Yanan Cao", "Kang Liu", "Jun Zhao"]
tags: ["LLM", "Planning", "Reasoning", "Task Decomposition", "Feedback Mechanism"]
institution: ["The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, CASIA", "School of Artificial Intelligence, University of Chinese Academy of Sciences", "Harbin Institute of Technology", "Institute of Information Engineering, Chinese Academy of Sciences", "School of Automation, Beijing Institute of Technology"]
description: "本文系统综述了大型语言模型在规划领域的应用现状，分类了外部模块增强、微调和搜索三大方法，总结了评估框架和挑战，为未来研究提供了理论指导和资源支持。"
---

> **Summary:** 本文系统综述了大型语言模型在规划领域的应用现状，分类了外部模块增强、微调和搜索三大方法，总结了评估框架和挑战，为未来研究提供了理论指导和资源支持。 

> **Keywords:** LLM, Planning, Reasoning, Task Decomposition, Feedback Mechanism

**Authors:** Pengfei Cao, Tianyi Men, Wencan Liu, Jingwen Zhang, Xuzhao Li, Xixun Lin, Dianbo Sui, Yanan Cao, Kang Liu, Jun Zhao

**Institution(s):** The Key Laboratory of Cognition and Decision Intelligence for Complex Systems, CASIA, School of Artificial Intelligence, University of Chinese Academy of Sciences, Harbin Institute of Technology, Institute of Information Engineering, Chinese Academy of Sciences, School of Automation, Beijing Institute of Technology


## Problem Background

大型语言模型（LLMs）在自然语言处理任务中表现出色，但其在规划（Planning）领域的应用潜力尚未被系统性探索。
规划作为智能体的核心能力，涉及环境理解、逻辑推理和顺序决策，而传统方法（如基于 PDDL 的符号规划）存在建模复杂性和鲁棒性不足的问题。
论文旨在通过综述现有研究，解决如何系统评估和分类 LLMs 在规划任务中的应用方法，以及识别当前挑战和未来方向的关键问题。

## Method

*   **外部模块增强方法（External Module Augmented Methods）**：
    *   **核心思想**：通过结合外部组件（如符号规划器、记忆模块）增强 LLMs 的规划能力，弥补其在复杂任务中的局限性。
    *   **具体实现**：包括 Planner Enhanced Methods（如 LLM+P 将自然语言转化为 PDDL 文件，再用经典规划器生成计划）和 Memory Enhanced Methods（如 MemoryBank 存储历史交互信息以支持长期规划）。
    *   **特点**：强调与传统规划工具的结合，适用于结构化任务，但可能增加计算成本。
*   **微调方法（Finetuning-based Methods）**：
    *   **核心思想**：通过轨迹数据或反馈信号调整 LLMs 参数，提升其对特定规划任务的适应性。
    *   **具体实现**：包括 Imitation Learning-based Methods（模仿专家或自生成轨迹）和 Feedback-based Methods（利用环境反馈、奖励模型或自我反思优化模型）。
    *   **特点**：注重模型自身学习能力，适合动态环境，但依赖高质量训练数据。
*   **搜索方法（Searching-based Methods）**：
    *   **核心思想**：通过任务分解、探索规划空间或优化解码策略，挖掘 LLMs 的推理潜力以寻找最优解。
    *   **具体实现**：包括 Decomposition-based Methods（将复杂任务分解为子任务）、Exploration-based Methods（如 Tree of Thoughts 使用树形结构探索推理路径）和 Decoding-based Methods（如 CoT-decoding 优化输出策略）。
    *   **特点**：适用于复杂推理任务，但时间复杂度较高，需平衡探索与效率。

## Experiment

*   **综述性质**：作为一篇综述性论文，未报告新的实验结果，而是总结了现有研究的评估框架，包括数据集（如 WebShop, ALFWorld）、评估指标（如 Success Rate, Execution Efficiency）和代表性方法的性能对比。
*   **方法效果**：外部模块增强方法在结构化任务中表现较好（如 PDDL 翻译准确性高），但计算成本较高；微调方法在动态环境下的适应性较强，但依赖数据质量；搜索方法在复杂推理任务中有效（如 ToT 提升推理深度），但效率较低。
*   **评估合理性**：数据集覆盖数字、实体、日常和垂直场景，较为全面，但评估指标多样化导致方法间直接比较困难；论文指出缺乏统一标准，影响了评估的客观性。
*   **局限性**：现有方法在复杂动态环境下的泛化能力和鲁棒性不足，部分方法（如搜索方法）在实际应用中面临计算开销问题。

## Further Thoughts

论文启发我思考如何将多模态数据（如视觉、文本）与 LLMs 的规划能力结合，特别是在实体场景（如机器人操作）中，视觉感知与语言推理的融合可能显著提升规划效果；此外，动态反馈机制（如结合强化学习与人类偏好对齐）或将成为提升 LLMs 自适应规划能力的关键；最后，构建统一评估框架以标准化规划能力测试，是推动领域发展的迫切需求。