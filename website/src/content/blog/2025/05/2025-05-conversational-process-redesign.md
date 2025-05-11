---
title: "Conversational Process Model Redesign"
pubDatetime: 2025-05-08T17:44:45+00:00
slug: "2025-05-conversational-process-redesign"
type: "arxiv"
id: "2505.05453"
score: 0.6147802048619904
author: "grok-3-latest"
authors: ["Nataliia Klievtsova", "Timotheus Kampik", "Juergen Mangler", "Stefanie Rinderle-Ma"]
tags: ["LLM", "Process Modeling", "Conversational AI", "Change Patterns", "Business Process Management"]
institution: ["Technical University of Munich", "SAP Signavio"]
description: "本文提出了一种对话式流程模型重设计方法，利用大型语言模型和变更模式赋能领域专家通过自然语言迭代修改流程模型，试图弥合沟通鸿沟，尽管在复杂模式应用和用户支持上仍有改进空间。"
---

> **Summary:** 本文提出了一种对话式流程模型重设计方法，利用大型语言模型和变更模式赋能领域专家通过自然语言迭代修改流程模型，试图弥合沟通鸿沟，尽管在复杂模式应用和用户支持上仍有改进空间。 

> **Keywords:** LLM, Process Modeling, Conversational AI, Change Patterns, Business Process Management

**Authors:** Nataliia Klievtsova, Timotheus Kampik, Juergen Mangler, Stefanie Rinderle-Ma

**Institution(s):** Technical University of Munich, SAP Signavio


## Problem Background

在业务流程管理（BPM）领域，领域专家与流程建模者之间的沟通鸿沟是一个长期存在的问题：领域专家缺乏建模技能，而建模者缺乏领域知识，导致流程模型的创建和重设计效率低下且成本高昂。
随着大型语言模型（LLM）的快速发展，利用自然语言交互赋能领域专家直接参与流程建模成为可能，本文旨在探索如何通过对话式流程模型重设计（Conversational Process Redesign, CPD）方法，让领域专家通过自然语言请求迭代地修改流程模型，同时解决模型正确性和完整性问题。

## Method

*   **核心思想：** 提出一种对话式流程模型重设计（CPD）方法，通过将用户的自然语言请求与文献中已有的流程变更模式（Change Patterns）结合，利用 LLM 实现流程模型的迭代修改。
*   **具体步骤：**
    *   **模式识别（Identify）：** LLM 分析用户输入的自然语言请求，尝试将其映射到预定义的变更模式（如插入流程片段、删除流程片段、并行化等），若无法匹配则返回失败提示。
    *   **含义推导（Derive）：** 若模式识别成功，LLM 将用户请求与识别出的变更模式结合，生成参数化的‘含义’（Meaning），即具体的修改指令，确保符合建模规则（如 BPMN 2.0）。
    *   **变更应用（Apply）：** 根据推导出的含义，LLM 对输入的流程模型进行修改，生成新的模型，并确保输出格式符合标准。
*   **模式扩展：** 基于对话式交互的复杂需求，论文对现有 14 个变更模式进行了分析，并提出了新的模式（如分割流程片段、合并流程片段、删除整个分支等），以覆盖更多用户请求场景。
*   **提示工程：** 通过设计系统指令和用户输入模板（Prompt Engineering），指导 LLM 的行为，确保其在模式识别、含义推导和变更应用中输出准确且符合预期。
*   **关键特点：** 该方法强调多步骤结构化处理，避免直接让 LLM 自由修改模型，而是通过变更模式提供可解释性和可重复性，同时支持用户与 LLM 的持续交互。

## Experiment

*   **有效性：** 在用户调研（64 名参与者）和三种 LLM（GPT-4o, Gemini-1.5-Pro, Mistral-Large-Latest）的测试中，仅 8 个变更模式（包括 4 个现有模式和 4 个新提出模式）在超过 30% 的案例中被成功实现，表明 LLM 在处理复杂变更模式时存在显著局限性。
*   **问题分析：** 失败案例中，9% 源于用户表述不清导致模式识别失败，12% 源于含义推导失败；此外，LLM 在应用复杂模式时的错误率较高，部分模式失败率高达 70%，显示出模式应用准确性的不足。
*   **模型对比：** GPT-4o 和 Gemini-1.5-Pro 在遵循指令和输出一致性上表现优于 Mistral-Large-Latest，但整体失败率分布相似，表明问题更多源于方法设计而非模型差异。
*   **实验设置：** 实验采用简单的 BPMN 模型以避免领域偏见，但可能限制了结果对复杂场景的适用性；用户未与 LLM 实时交互，而是基于静态输入-输出对提供表述，可能与真实交互场景有偏差；此外，单次提示限制了用户澄清意图的机会。
*   **结论：** 实验表明，CPD 方法在简单模式上具有一定可行性，但需要在用户表述支持和复杂模式应用准确性上进一步改进。

## Further Thoughts

论文提出将 LLM 与传统确定性方法结合的思路非常具有启发性，例如 LLM 可用于模式识别和参数提取，而模式应用通过算法实现以提高准确性；此外，用户行为模式（如‘从头创建模型’或‘回退到前一状态’）作为交互需求，启发我们可以在对话式 AI 中引入推荐系统或动态提示机制，减少用户表述歧义，提升交互效率；更进一步，可以探索如何通过对话式交互逐步培养领域专家的‘流程思维’，将其建模技能提升作为长期目标。