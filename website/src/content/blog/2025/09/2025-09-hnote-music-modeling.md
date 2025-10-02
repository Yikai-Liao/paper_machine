---
title: "HNote: Extending YNote with Hexadecimal Encoding for Fine-Tuning LLMs in Music Modeling"
pubDatetime: 2025-09-30T02:50:01+00:00
slug: "2025-09-hnote-music-modeling"
type: "arxiv"
id: "2509.25694"
score: 0.8007608148303644
author: "grok-3-latest"
authors: ["Hung-Ying Chu", "Shao-Yu Wei", "Guan-Wei Chen", "Tzu-Wei Hung", "Cheng Yang Tsai", "Yu-Cheng Lin"]
tags: ["LLM", "Music Notation", "Fine-Tuning", "Symbolic Representation", "Structural Alignment"]
institution: ["Yuan Ze University"]
description: "本文提出HNote，一种基于十六进制的音乐符号表示系统，通过固定32单位节拍结构和统一编码提升LLM在音乐建模中的生成质量，并在江南风格音乐上取得显著效果。"
---

> **Summary:** 本文提出HNote，一种基于十六进制的音乐符号表示系统，通过固定32单位节拍结构和统一编码提升LLM在音乐建模中的生成质量，并在江南风格音乐上取得显著效果。 

> **Keywords:** LLM, Music Notation, Fine-Tuning, Symbolic Representation, Structural Alignment

**Authors:** Hung-Ying Chu, Shao-Yu Wei, Guan-Wei Chen, Tzu-Wei Hung, Cheng Yang Tsai, Yu-Cheng Lin

**Institution(s):** Yuan Ze University


## Problem Background

大型语言模型（LLMs）在符号化音乐生成中展现出潜力，但传统音乐表示格式（如MIDI, MusicXML, ABC Notation）因复杂性高、结构不一致或冗长而不适合直接用于LLM训练。
YNote虽提供简化表示，但缺乏固定节拍对齐机制，导致模型难以学习稳定的节奏结构。
本文提出HNote，旨在设计一种简洁、结构一致的符号表示系统，解决对齐问题并提升LLM在音乐建模中的生成质量。

## Method

*   **核心思想:** 提出HNote，一种基于十六进制的音乐符号表示系统，通过扩展YNote，引入固定32单位节拍结构和统一编码方式，确保节奏对齐和结构一致性，适配LLM训练。
*   **具体设计:** 
    *   使用两digit十六进制编码表示音高（Pitch Onset, '00'到'7F'）和持续时间（Note Duration, '80'到'FF'），形成紧凑且统一的词汇表。
    *   每个小节固定为32单位长度，确保不同节奏值（如四分音符、八分音符）在同一框架内精确对齐，消除节奏不一致问题。
*   **数据处理:** 将12,300首江南风格歌曲从YNote转换为HNote，通过自动化转换流程映射音高和持续时间，确保数据集一致性。
*   **模型训练:** 基于LLaMA-3.1 (8B)模型，采用LoRA（Low-Rank Adaptation）进行参数高效微调，降低计算成本，并在推理时通过指定首尾音符作为软约束，增强生成音乐的风格一致性。
*   **关键优势:** HNote在保持简洁性和可读性的同时，通过结构化设计提升了LLM对音乐序列模式的学习能力。

## Experiment

*   **有效性:** HNote生成的音乐在语法正确性上达到82.5%，表明其作为符号框架的可靠性；BLEU和ROUGE评估显示生成音乐与参考音乐在符号和结构上高度相似，尤其ROUGE-L分数较高（如数据集内最高0.625），表明全局结构一致性较好。
*   **实验设置:** 实验分为两组，一组从数据集中随机选取5首江南风格歌曲，另一组选取5首未包含在训练集中的外部江南歌曲，通过首尾音符提示生成音乐，并计算BLEU-1到BLEU-4、ROUGE-1、ROUGE-2和ROUGE-L分数，结果在两组中一致，显示出较好的泛化能力。
*   **局限性:** 实验主要集中于江南风格音乐，数据集风格单一，可能限制模型在其他音乐风格上的表现；错误主要来自解码不完整或音高起始符号缺失，提示需优化解码策略。
*   **总体评价:** HNote相较于传统格式在结构一致性和风格保持上表现显著优越，为LLM在音乐建模中的应用提供了可靠基础。

## Further Thoughts

HNote的固定节拍结构设计为其他序列建模任务提供了启发，例如在自然语言处理中引入固定长度单元以增强对长序列结构的理解；此外，十六进制编码的紧凑性提示可以在图像或视频序列建模中探索类似表示方法；未来是否可以通过扩展HNote表示（如加入和弦、动态标记）或结合多模态数据（如文本与音乐符号）进一步提升音乐生成系统的表达能力和复杂性？