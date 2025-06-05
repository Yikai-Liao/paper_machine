---
title: "Performance of leading large language models in May 2025 in Membership of the Royal College of General Practitioners-style examination questions: a cross-sectional analysis"
pubDatetime: 2025-06-03T15:25:38+00:00
slug: "2025-06-llm-primary-care-exam"
type: "arxiv"
id: "2506.02987"
score: 0.5899876600650569
author: "grok-3-latest"
authors: ["Richard C Armitage"]
tags: ["LLM", "Reasoning", "Medical Education", "Primary Care", "Clinical Decision"]
institution: ["University of Nottingham"]
description: "本文首次测试了 2025 年领先推理模型在皇家全科医生学会（MRCGP）风格考试中的表现，证明其在初级医疗教育中显著超越人类平均水平，展现出支持临床实践的巨大潜力。"
---

> **Summary:** 本文首次测试了 2025 年领先推理模型在皇家全科医生学会（MRCGP）风格考试中的表现，证明其在初级医疗教育中显著超越人类平均水平，展现出支持临床实践的巨大潜力。 

> **Keywords:** LLM, Reasoning, Medical Education, Primary Care, Clinical Decision

**Authors:** Richard C Armitage

**Institution(s):** University of Nottingham


## Problem Background

大型语言模型（LLMs）在临床医学领域展现出巨大潜力，但现有研究多集中于 ChatGPT-4 及其前身，而对 2025 年最新、最强大的‘推理模型’（reasoning models）在初级医疗（primary care）教育中的表现研究不足。
本文旨在测试四款领先 LLMs（OpenAI 的 o3、Anthropic 的 Claude Opus 4、xAI 的 Grok-3 和 Google 的 Gemini 2.5 Pro）在皇家全科医生学会（MRCGP）风格考试题目中的能力，解决推理模型在初级医疗知识和复杂推理任务中的应用潜力这一关键问题。

## Method

*   **研究设计:** 采用跨 sectional 研究设计，于 2025 年 5 月 25 日测试四款领先 LLMs 的表现。
*   **测试内容:** 从皇家全科医生学会（RCGP）的 GP SelfTest 工具中通过‘Lucky Dip’功能随机抽取 100 道多选题，涵盖文本信息、实验室结果和临床图像。
*   **提示方式:** 每款模型被提示以英国全科医生（GP）的身份回答问题，完整的问题信息（包括文本和图像）被输入到模型的上下文窗口中，无额外提示。
*   **测试流程:** 题目按顺序呈现，每道题每款模型仅尝试一次，答案根据 GP SelfTest 提供的正确答案评分（正确得 1 分，错误得 0 分），最终计算每款模型的总得分百分比。
*   **目标:** 通过模拟真实考试环境，评估模型在初级医疗知识、临床推理和多模态数据处理能力上的表现。

## Experiment

*   **有效性:** 四款模型表现均非常出色，o3 得分为 99.0%，Claude Opus 4、Grok-3 和 Gemini 2.5 Pro 均为 95.0%，显著超越同行（GPs 和 GP 实习医生）平均得分 73.0%。
*   **差异性:** o3 表现最佳，仅答错 1 题，其他模型各错 5 题；所有模型在正确和错误答案中表现出相同自信度，暴露出对不确定性表达的不足。
*   **实验设置:** 随机抽取题目和单一尝试的设计合理，模拟真实考试情境；但题目数量（100 道）较少，且非文本数据（如图像、表格）比例低（仅 6 道题），可能限制了对模型多模态能力的全面评估。
*   **结论:** 实验证明推理模型在初级医疗教育中的巨大潜力，但需关注不确定性表达和更广泛测试的必要性。

## Further Thoughts

推理模型通过多步推理和链式思考显著提升复杂任务表现，未来可探索更多推理增强技术在医疗教育中的应用；模型对不确定性的表达不足，提示需研究如何融入不确定性评估以提高临床安全性；此外，针对初级医疗数据训练的领域特定模型可能表现更优，启发我们思考如何结合本地化临床指南和最新研究进一步优化 LLMs 在垂直领域的应用。