---
title: "TRUST: An LLM-Based Dialogue System for Trauma Understanding and Structured Assessments"
pubDatetime: 2025-05-01T18:25:27+08:00
slug: "2025-04-trust-ptsd-dialogue"
score: 0.8697280710230166
author: "grok-3-mini-latest"
authors: ["Sichang Tu", "Abigail Powers", "Stephen Doogan", "Jinho D. Choi"]
tags: ["LLM", "Dialogue System", "Diagnostic Interview", "Patient Simulation", "Mental Health"]
institution: ["Emory University", "DooGood Foundation"]
description: "本文提出TRUST框架，利用LLMs和自定义Dialogue Acts schema构建一个模拟临床医生的对话系统，进行PTSD结构化诊断访谈，并通过基于真实转录的患者模拟进行评估，展现了与真实访谈相当的性能。"
---

> **Summary:** 本文提出TRUST框架，利用LLMs和自定义Dialogue Acts schema构建一个模拟临床医生的对话系统，进行PTSD结构化诊断访谈，并通过基于真实转录的患者模拟进行评估，展现了与真实访谈相当的性能。 

> **Keywords:** LLM, Dialogue System, Diagnostic Interview, Patient Simulation, Mental Health
> **Recommendation Score:** 0.8697280710230166

**Authors:** Sichang Tu, Abigail Powers, Stephen Doogan, Jinho D. Choi
**Institution(s):** Emory University, DooGood Foundation

## Problem Background

这篇论文的出发点是解决心理健康护理的可及性问题，特别是针对创伤后应激障碍（PTSD）的诊断挑战。背景在于，美国有超过2800万成年人有心理健康问题但未得到治疗，其中许多人无法进行正式诊断，因为心理健康提供者短缺、医疗成本高昂，以及PTSD评估的复杂性（如需要遵循DSM-5标准的结构化访谈）。论文指出，虽然大型语言模型（LLMs）已在心理健康对话系统中被探索用于检测、干预和咨询，但鲜有研究专注于正式的诊断访谈系统，尤其是针对PTSD的长时间结构化评估。因此，本文的关键问题是如何开发一个LLM驱动的对话系统，来模拟临床医生的行为，桥接心理健康护理的差距，实现更高效、成本更低的PTSD诊断。

## Method

*   **核心思想:** 论文提出TRUST框架，这是一个基于LLMs的合作模块系统，旨在模拟临床医生进行PTSD诊断访谈。核心是通过一个自定义的Dialogue Acts (DA) schema作为中间层，提高对话的控制性和结构化，同时确保自然流畅。
*   **具体实现:** 系统包括Database模块（存储变量、历史记录和评估分数）和Framework模块（包含Conversation和Assessment子模块）。具体步骤如下：
    - **Database模块:** 初始化变量（包括变量依赖、面试问题树结构等）、历史记录（对话轮次、DA标签等）和分数（评估结果）。变量被分类为独立或依赖类型，问题分为核心和可选。
    - **Framework模块:** Conversation子模块使用LLM（如Claude）预测下一个响应的DA标签（8个标签：GC为问候/结束、GI为指导/说明、ACK为确认、EMP为移情/支持、VAL为验证、IS为信息寻求、CQ为澄清问题、CA为澄清回答），然后基于标签生成响应或选择问题；Assessment子模块根据对话历史进行诊断评估。整个过程通过决策流程（如步骤①-⑦）动态管理对话流，例如预测DA标签后生成响应，或根据患者输入决定是否继续提问。
    - **患者模拟:** 为了测试系统，使用真实访谈转录作为基础，通过LLM生成模拟患者响应，确保响应真实性和一致性，避免幻觉。
*   **关键特点:** 该方法不直接修改LLMs，而是通过DA schema和模块化设计增强控制，避免了端到端系统的风险（如生成不适当响应），并支持扩展到其他心理健康条件。

## Experiment

*   **有效性:** 实验使用Tu et al.的PTSD访谈数据集，采样100个转录进行评估。人类专家（对话专家和临床专家）通过5点Likert量表评估，显示TRUST在Comprehensiveness（症状探索深度）、Appropriateness（临床有效性）和Communication Style（互动质量）上与真实临床访谈相当，平均得分接近0（例如，Comprehensiveness得分0.03和-0.14，表示等效性能）。患者模拟在Completeness和Appropriateness上得分可接受（平均0.32和0.31），但Faithfulness（忠实度）较低（-0.39和-0.33），表明LLM偶尔会引入幻觉或遗漏信息。
*   **提升是否明显:** 方法提升明显，因为TRUST在保持结构化诊断的同时，实现了与真实访谈相当的性能，而LLMs评估（如Claude和GPT）显示过度乐观（平均1.68和1.76），突显人类评估的必要性。相比传统访谈，系统减少了时间和成本开销。
*   **实验设置是否全面合理:** 实验设置全面，包括症状级评估（将转录分组为25个PTSD症状集群）、配对比较（生成 vs. 原转录）和多指标评估（人类和LLM评估）。合理性高，因为它使用真实数据避免偏差，并考虑了LLM的局限性（如幻觉问题），尽管LLM评估显示不一致性。

## Further Thoughts

论文中值得关注的启发性想法是，使用Dialogue Acts schema作为中间层来控制LLM对话系统的决策过程，这可以推广到其他领域，如教育或医疗训练中，以提高AI在复杂任务中的可靠性和适应性；此外，基于真实数据的患者模拟方法提示了如何利用LLM进行大规模模拟训练，而非依赖手动测试，这可能启发未来开发更鲁棒的AI模拟系统来处理情感和临床 nuance。