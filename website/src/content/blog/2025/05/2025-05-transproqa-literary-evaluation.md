---
title: "TransProQA: an LLM-based literary Translation evaluation metric with Professional Question Answering"
pubDatetime: 2025-05-08T17:12:56+00:00
slug: "2025-05-transproqa-literary-evaluation"
type: "arxiv"
id: "2505.05423"
score: 0.615135203330216
author: "grok-3-latest"
authors: ["Ran Zhang", "Wei Zhao", "Lieve Macken", "Steffen Eger"]
tags: ["LLM", "Literary Translation", "Evaluation Metric", "Question Answering", "Cultural Adaptation"]
institution: ["University of Mannheim", "University of Aberdeen", "University of Gent", "University of Technology Nuremberg", "Natural Language Learning and Generation Lab"]
description: "本文提出 TRANS PRO QA，一种基于 LLM 的问答框架，通过融入专业翻译者视角评估文学翻译质量，显著提升了与人类判断的相关性和充分性，超越现有 SOTA 指标。"
---

> **Summary:** 本文提出 TRANS PRO QA，一种基于 LLM 的问答框架，通过融入专业翻译者视角评估文学翻译质量，显著提升了与人类判断的相关性和充分性，超越现有 SOTA 指标。 

> **Keywords:** LLM, Literary Translation, Evaluation Metric, Question Answering, Cultural Adaptation

**Authors:** Ran Zhang, Wei Zhao, Lieve Macken, Steffen Eger

**Institution(s):** University of Mannheim, University of Aberdeen, University of Gent, University of Technology Nuremberg, Natural Language Learning and Generation Lab


## Problem Background

大型语言模型（LLMs）在文学翻译等创意领域的应用日益广泛，但现有自动评估指标（如 BLEU, BERTScore, XCOMET）主要关注语义准确性和语言流畅性，忽视了文学翻译中关键的文学特质，包括文化背景、文学修辞和作者风格，导致机器翻译（MT）常被高估，而专业人类翻译的价值被低估，长期可能损害翻译质量和文化真实性。
论文旨在开发一个专门针对文学翻译的评估指标，解决现有指标无法捕捉文学细微差别、缺乏专业翻译者视角的问题。

## Method

*   **微调 XCOMET-XL：** 针对现有指标在文学领域的不足，论文尝试通过在文学数据集（如 PAR3-UNANNOTATED 和 WMT24）上进行微调，适应文学翻译评估需求。具体包括：
    *   使用排名任务，通过三元组损失（Triplet Loss）调整模型嵌入空间，使人类翻译与源文本更接近，机器翻译更远。
    *   使用回归任务，通过均方误差损失（MSE Loss）使模型预测的质量分数与人类标注分数对齐。
    *   选择性微调模型顶层（1/4 或 1/2 层），以保留通用能力同时适应文学领域，测试不同数据集配置（如单语言对 Fr-En 和多语言对 XX-En）。
*   **TRANS PRO QA 框架：** 提出一个全新的、无训练、参考无关的 LLM 问答（QA）评估指标，模拟专业翻译者的质量控制过程，核心步骤如下：
    *   **问题列表设计：** 从文学翻译理论、实践资源和专业翻译者访谈中收集 45 个问题，覆盖语法、文学设备、文化适应、语气一致性等多个方面；通过专业翻译者投票（评分 1-5）和 LLM 敏感性分析（排除区分度低的问题），筛选出 25 个关键问题。
    *   **问答评估过程：** 使用 LLM（如 GPT-4o-mini, LLaMA3.3-70b）对源文本和翻译文本回答这些问题，答案为 'Yes'、'No' 或 'Maybe'，分别映射为分数 1、0、0.5；最终翻译分数为问题的平均分，可根据翻译者投票加权。
    *   **提示模板优化：** 设计多种提示策略，包括基础版（Vanilla，仅简单指令）、提示级逐步指令（PromptStep，提供详细评估步骤）和问题级逐步指令（QuestionStep，针对每个问题提供具体指导），以测试指令复杂性对评估效果的影响。
*   **核心创新：** TRANS PRO QA 通过结构化问答形式，将专业翻译者的评估标准融入 LLM 评估中，捕捉文学翻译的细微差别，同时避免了传统参考依赖方法的局限性。

## Experiment

*   **微调 XCOMET-XL 效果：** 在 LIT EVAL-CORPUS 数据集上，1/4-FREN 变体在相关性指标 ACC-EQ 和 Kendall’s τ 上略有提升（分别从 0.528 到 0.542 和 0.387 到 0.406），1/2-FREN 在充分性（Human > top systems）上达到 29.4%，但整体提升有限且不一致；在 LITERARY TRAN 和 PAR3-ANNOTATED 数据集上，部分配置甚至导致性能下降，充分性平均提升仅 1-3 个百分点，显示微调策略对文学领域的适应性有限。
*   **TRANS PRO QA 效果：** 在所有三个数据集上显著优于基线和微调方法。以 LIT EVAL-CORPUS 为例，Vanilla w 变体在 ACC-EQ 和 Kendall’s τ 上分别达到 0.616 和 0.605，相比最佳 SOTA 提升约 0.06 和 0.035；在充分性方面，Human > top systems 提升至 41.4%，比最佳 SOTA 高 14.7 个百分点，Human > all systems 和 Human > all but top systems 分别提升 16.4% 和 22.5%。在 LITERARY TRAN 上，充分性提升 16.4% 至 42.4%；在 PAR3-ANNOTATED 上，相关性提升 0.08，充分性提升 17.8 个百分点。
*   **实验设置合理性：** 数据集涵盖多种语言对（高资源为主）和文学类型（当代与经典），评估了不同模型（闭源如 GPT-4o-mini，开源如 LLaMA3.3-70b）和多种提示策略，充分性测试考虑了不同难度的对比（如与顶级 MT 系统对比），设置全面且合理；不足之处在于缺乏低资源语言对的覆盖和对长篇叙事单位的评估。
*   **额外观察：** 翻译者投票加权进一步提升了性能（如 LIT EVAL-CORPUS 上 ACC-EQ 提升 0.01-0.05），TRANS PRO QA 在开源模型上也表现出色，显示其普适性和低依赖性。

## Further Thoughts

TRANS PRO QA 通过 QA 形式将人类专家的评估标准结构化地融入 LLM 评估，这一思路启发我们可以在其他主观性强的领域（如诗歌生成、叙事创作）设计领域特定问题列表，提升评估的细粒度和对齐度；此外，翻译者投票加权的机制提示未来可以探索动态权重调整或多轮专家反馈机制，以进一步逼近人类评估水平；最后，其无训练特性表明提示工程可能在特定领域评估中比微调更高效，值得研究如何通过精心设计的提示框架最大化利用现有模型能力。