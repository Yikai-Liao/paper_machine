---
title: "SpeechR: A Benchmark for Speech Reasoning in Large Audio-Language Models"
pubDatetime: 2025-08-04T03:28:04+00:00
slug: "2025-08-speech-reasoning-benchmark"
type: "arxiv"
id: "2508.02018"
score: 0.6942791025589355
author: "grok-3-latest"
authors: ["Wanqi Yang", "Yanda Li", "Yunchao Wei", "Meng Fang", "Ling Chen"]
tags: ["LLM", "Audio Language Model", "Speech Reasoning", "Factual Reasoning", "Procedural Reasoning", "Normative Reasoning"]
institution: ["University of Technology Sydney", "University of Liverpool", "Beijing Jiaotong University"]
description: "SpeechR 是一个专为大型音频-语言模型设计的语音推理基准测试，系统评估事实性、程序性和规范性推理能力，并探索声学特征影响，揭示当前模型局限并推动多模态推理研究。"
---

> **Summary:** SpeechR 是一个专为大型音频-语言模型设计的语音推理基准测试，系统评估事实性、程序性和规范性推理能力，并探索声学特征影响，揭示当前模型局限并推动多模态推理研究。 

> **Keywords:** LLM, Audio Language Model, Speech Reasoning, Factual Reasoning, Procedural Reasoning, Normative Reasoning

**Authors:** Wanqi Yang, Yanda Li, Yunchao Wei, Meng Fang, Ling Chen

**Institution(s):** University of Technology Sydney, University of Liverpool, Beijing Jiaotong University


## Problem Background

大型音频-语言模型（LALMs）在语音转录和情感识别等表面感知任务上表现出色，但其在语音场景中的上下文推理和复杂推理能力尚未得到充分评估。
现有基准测试主要关注低层次感知任务，缺乏对事实性、程序性和规范性推理的系统性考察，也未充分研究声学特征（如情感和重音）对推理的影响。

## Method

*   **核心思想:** 构建一个统一的语音推理基准测试 SpeechR，评估 LALMs 在不同类型推理任务上的表现，并探索声学特征的影响。
*   **数据构建:** 基于高质量文本推理数据集（如 ReClor, GSM8K, ETHICS），通过 Azure Speech SDK 合成语音数据，确保语义和认知意图一致，同时控制语音内容、语调和结构。
*   **推理类型设计:** 包含三种推理类型：
    *   事实性推理（Factual Reasoning）：涉及具体信息检索和常识理解，不需多步推理。
    *   程序性推理（Procedural Reasoning）：要求多步逻辑或数值推理，如数学问题解决。
    *   规范性推理（Normative Reasoning）：基于社会、伦理或行为规范进行判断，常涉及主观评估。
*   **评估格式:** 提供三种版本以支持多维度评估：
    *   多选版本（Multiple-Choice Version）：标准化选项，评估答案选择准确率。
    *   生成版本（Generative Version）：要求模型生成自由形式推理链，评估逻辑一致性和连贯性。
    *   声学特征版本（Acoustic-Feature Version）：引入情感和重音变化，研究声学因素对推理的影响。
*   **质量控制:** 通过多阶段验证确保基准测试可靠性，包括文本准确性、文本-音频对齐和音频自然度评估（通过人类听觉测试，平均自然度评分达 4.8/5）。

## Experiment

*   **有效性:** 在多选版本中，顶级模型（如 Gemini-1.5-Pro）在事实性任务上表现较好（准确率超 70%），但在程序性和规范性任务上准确率显著下降，尤其在道德判断等社会推理任务中；生成版本中，仅少数模型（如 GPT-4o）在程序性任务上展现多步推理能力，大多数模型逻辑连贯性较差。
*   **声学影响:** 声学特征版本显示情感和重音变化对推理表现有一定影响，部分模型（如 Mellow）在情感增强下表现提升，而大型模型（如 GPT-4o）在重音变化下略有下降，表明模型对语音表达的敏感性差异。
*   **全面性与合理性:** 实验覆盖 11 个先进 LALMs，包含开源和专有模型，评估维度包括准确率、生成质量和声学鲁棒性，设置较为全面；但与文本任务相比，语音任务准确率普遍下降，凸显多模态对齐和上下文理解的挑战。
*   **局限性:** 实验数据为合成语音，可能无法完全反映真实语音场景的复杂性；此外，规范性推理任务的主观性可能导致评估偏差。

## Further Thoughts

SpeechR 通过文本到语音转换构建语音推理任务，为多模态基准测试提供了可扩展思路，未来是否可以引入真实语音数据以模拟更复杂的现实场景？
声学特征对推理的影响值得深入探索，是否可以通过设计专门的多模态融合架构，让模型更好地利用情感和重音线索提升推理能力？
规范性推理的不足提示我们，是否应引入更多文化和语境多样性的训练数据，以增强模型在社会和道德判断任务上的表现？