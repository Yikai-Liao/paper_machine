---
title: "VoxRole: A Comprehensive Benchmark for Evaluating Speech-Based Role-Playing Agents"
pubDatetime: 2025-09-04T07:03:46+00:00
slug: "2025-09-voxrole-speech-roleplay"
type: "arxiv"
id: "2509.03940"
score: 0.7653122016708404
author: "grok-3-latest"
authors: ["Weihao Wu", "Liang Cao", "Xinyu Wu", "Zhiwei Lin", "Rui Niu", "Jingbei Li", "Zhiyong Wu"]
tags: ["LLM", "Role-Playing", "Speech Synthesis", "Persona Consistency", "Evaluation Benchmark"]
institution: ["Shenzhen International Graduation School, Tsinghua University", "StepFun"]
description: "本文提出 VoxRole 基准，通过自动化提取电影中的语音对话和多维角色画像，为语音角色扮演对话代理的评估提供了首个全面框架，并揭示了当前模型在副语言表达和角色一致性上的不足。"
---

> **Summary:** 本文提出 VoxRole 基准，通过自动化提取电影中的语音对话和多维角色画像，为语音角色扮演对话代理的评估提供了首个全面框架，并揭示了当前模型在副语言表达和角色一致性上的不足。 

> **Keywords:** LLM, Role-Playing, Speech Synthesis, Persona Consistency, Evaluation Benchmark

**Authors:** Weihao Wu, Liang Cao, Xinyu Wu, Zhiwei Lin, Rui Niu, Jingbei Li, Zhiyong Wu

**Institution(s):** Shenzhen International Graduation School, Tsinghua University, StepFun


## Problem Background

大型语言模型（LLMs）的进步推动了角色扮演对话代理（RPCAs）的发展，但当前研究主要集中于文本模态，忽略了语音中的副语言特征（如语调、韵律、节奏），这些特征对塑造角色情感和身份至关重要；此外，现有语音对话数据集缺乏标准化的评估基准，尤其是针对角色一致性等核心能力的评估，导致无法有效量化模型在沉浸式角色扮演中的表现。

## Method

* **核心思想：** 构建一个全面的基准 VoxRole，专门用于评估基于语音的角色扮演对话代理，通过自动化流程从电影中提取语音对话数据并构建多维角色画像。
* **具体实现：**
  * **第一阶段 - 语音对话提取：** 从电影脚本和音频数据出发，通过数据收集与准备（提取音频、解析脚本）、字级音频-脚本对齐（使用去噪模型 Resemble、转录工具 Whisper 和强制对齐工具 Wav2Vec2.0，结合动态匹配算法计算最小编辑距离）和语义验证与对话整理（使用 MPNet 计算语义相似度，筛选连续多轮对话），生成带有发言者标注的高质量对话数据。
  * **第二阶段 - 角色画像提炼（Persona Distillation）：** 使用大型语言模型（LLM）从剧本中提取角色的多维度特征，包括个性（基于场景事件总结推断）、语言风格（直接分析角色对话）、人际关系（基于共同事件历史）和声学特征（提取平均音高、能量和语速并分类为高、中、低）。最终将这些特征整合为第二人称叙事，作为角色扮演的提示。
* **评估框架：** 设计了双重评估方法，包括基于指标的评估（Rouge-L, Meteor, BertScore F1 衡量文本相似度，UTMOS 评估语音质量）和基于 LLM 的评估（结合文本和声学特征，评估人性化、个性一致性、语言忠诚度、关系连贯性、上下文连贯性和副语言适当性等维度）。
* **关键点：** 整个流程完全自动化，避免了手动标注的高成本和主观性，具有较高的可扩展性，同时注重语音模态对角色扮演沉浸感的贡献。

## Experiment

* **有效性：** 实验评估了多个主流语音对话模型（包括开源模型如 Qwen2.5-Omni、GLM-4-Voice 和闭源模型如 GPT-4o），结果显示 GPT-4o 在文本生成（Rouge-L 12.91, Meteor 18.30）和语音质量（UTMOS 3.66）上表现最佳，而开源模型 Qwen2.5-Omni（7B 参数）在语音自然度上接近 GPT-4o（UTMOS 3.57），表明性能不完全依赖模型规模。
* **全面性：** 实验设置涵盖了词汇和语义相似度、语音质量以及角色扮演特有维度（如个性一致性、关系连贯性）的评估，结合 LLM 评估和人工验证（Pearson 相关系数 0.762），确保了结果的可靠性和多维度覆盖。
* **局限性揭示：** 所有模型在副语言适当性（Acoustic Quality）上表现较弱，即使是 GPT-4o（评分 3.82）也远低于其在其他维度的表现；开源模型在个性一致性和关系理解上与闭源模型差距明显（超过 15%）。
* **消融实验：** 上下文长度实验表明，最优上下文窗口存在（长度为 6 时性能最佳），过短或过长上下文均会影响角色扮演能力。

## Further Thoughts

本文启发我思考语音模态在角色扮演中的核心作用，副语言特征不仅是对话的补充，而是塑造角色身份和情感深度的关键，未来可以通过更精细的声学特征建模进一步提升沉浸感；此外，自动化角色画像提取的思路可以扩展到其他多模态场景（如游戏、虚拟现实），甚至引入动态角色发展机制，模拟随对话进展而变化的情感和态度，以更真实地接近人类交互。