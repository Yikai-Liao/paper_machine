---
title: "AI-Driven Scholarly Peer Review via Persistent Workflow Prompting, Meta-Prompting, and Meta-Reasoning"
pubDatetime: 2025-05-06T09:06:18+00:00
slug: "2025-05-persistent-workflow-prompting"
type: "arxiv"
id: "2505.03332"
score: 0.4817256272239449
author: "grok-3-latest"
authors: ["Evgeny Markhasin"]
tags: ["LLM", "Prompt Engineering", "Workflow Design", "Reasoning", "Bias Mitigation"]
institution: ["Lobachevsky State University of Nizhny Novgorod"]
description: "本文提出持久工作流程提示（PWP）方法，通过结构化提示库和元提示技术，指导大型语言模型完成复杂的学术同行评审任务，并在抑制输入偏见方面取得初步成功。"
---

> **Summary:** 本文提出持久工作流程提示（PWP）方法，通过结构化提示库和元提示技术，指导大型语言模型完成复杂的学术同行评审任务，并在抑制输入偏见方面取得初步成功。 

> **Keywords:** LLM, Prompt Engineering, Workflow Design, Reasoning, Bias Mitigation

**Authors:** Evgeny Markhasin

**Institution(s):** Lobachevsky State University of Nizhny Novgorod


## Problem Background

大型语言模型（LLMs）在处理学术同行评审等复杂专家任务时，面临数据限制和推理复杂性的挑战，尤其是在需要深度批判性分析和克服输入偏见（Input Bias）的情况下。
本文旨在探索如何通过提示工程技术，在不依赖API或代码的情况下，仅使用标准聊天界面，将专家评审的隐性知识和工作流程转化为结构化指导，从而让 LLMs 能够系统化地完成学术评审任务。

## Method

*   **核心思想：** 提出持久工作流程提示（Persistent Workflow Prompting, PWP），通过一个层次化、模块化的提示架构，指导 LLMs 完成复杂多步骤分析任务，如学术同行评审。
*   **具体实现：**
    *   **PWP 架构：** 使用 Markdown 格式组织详细的工作流程，形成一个持久的工作流程库，在会话开始时一次性提交到 LLMs 的上下文内存中。随后通过简短用户查询触发具体分析步骤，避免重复提交大型提示，节省上下文窗口空间。
    *   **工作流程设计：** 将评审任务分解为可管理的步骤（如识别主要结果、评估方法论、定量可行性检查等），并编码为提示中的具体指令，指导 LLMs 进行系统化分析。
    *   **角色设计（Persona Engineering）：** 通过详细的角色描述和行为指令，赋予 LLMs 批判性评审者的特质（如怀疑精神、客观性），以对抗模型固有的正向输入偏见，强调独立方法评估。
    *   **元提示（Meta-Prompting）：** 利用 LLMs 自身迭代优化提示内容，包括语言结构优化和语义工作流程设计，通过与模型的交互逐步提炼 PWP 提示。
    *   **元推理（Meta-Reasoning）：** 通过反思专家评审的隐性知识和直觉判断，将其转化为明确的提示指令，例如将‘过于美好而不真实’的启发式判断分解为具体的可操作检查步骤。
*   **关键特点：** 不修改底层模型，仅通过推理时提示调整实现复杂任务指导，适用于大多数现成 LLMs，包括专有模型。

## Experiment

*   **有效性：** 在针对实验化学论文的测试中，PWP 指导下的 LLMs（如 Google Gemini Advanced 2.5 Pro）能够一致地识别出单一测试案例中的主要方法论缺陷，并在不同模型（如 ChatGPT o1/o3、SuperGrok Grok 3 Think）间表现出一定的分析稳定性，成功抑制了输入偏见。
*   **优越性：** 相比简单的提示方法，PWP 通过结构化工作流程显著提升了 LLMs 的批判性分析深度，尤其在多模态分析（结合文本和图像）方面，Gemini Advanced 2.5 Pro 甚至发现了人类评审未注意到的缺陷。
*   **实验设置局限：** 实验仅基于单一测试案例（已知有方法论缺陷的化学论文），缺乏多案例验证和量化基准，提示范围局限于核心实验方法，未覆盖论文其他部分（如数据呈现、统计分析），因此通用性和全面性有待进一步验证。
*   **开销与兼容性：** PWP 提示体积较大（超过 30 kB），可能超出某些 LLM 聊天界面的输入限制（如 Qwen 界面），对平台兼容性构成挑战。

## Further Thoughts

PWP 的持久工作流程库理念可以扩展到其他复杂任务领域，如实验设计、代码审查或跨学科分析，构建领域特定的提示库；此外，元提示和元推理技术启发了一种自适应提示开发框架，可以利用 LLMs 自身迭代优化提示设计，甚至形成自动化提示改进循环；同时，输入偏见抑制策略（通过负向偏见角色设计）也为在教育或决策支持系统中平衡 LLMs 的学习能力和批判性思维提供了思路。