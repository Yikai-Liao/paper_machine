---
title: "CardiffNLP at CLEARS-2025: Prompting Large Language Models for Plain Language and Easy-to-Read Text Rewriting"
pubDatetime: 2025-08-05T09:16:19+00:00
slug: "2025-08-text-simplification-prompting"
type: "arxiv"
id: "2508.03240"
score: 0.669995844829963
author: "grok-3-latest"
authors: ["Mutaz Ayesh", "Nicolás Gutiérrez-Rolón", "Fernando Alva-Manchego"]
tags: ["LLM", "Text Simplification", "Prompt Engineering", "Accessibility", "Evaluation Metrics"]
institution: ["Cardiff University"]
description: "本文通过提示工程利用大型语言模型实现西班牙语文本的自动化简化为 Plain Language 和 Easy-to-Read 格式，在 CLEARS-2025 任务中取得优异成绩，为信息可访问性提供了有效工具。"
---

> **Summary:** 本文通过提示工程利用大型语言模型实现西班牙语文本的自动化简化为 Plain Language 和 Easy-to-Read 格式，在 CLEARS-2025 任务中取得优异成绩，为信息可访问性提供了有效工具。 

> **Keywords:** LLM, Text Simplification, Prompt Engineering, Accessibility, Evaluation Metrics

**Authors:** Mutaz Ayesh, Nicolás Gutiérrez-Rolón, Fernando Alva-Manchego

**Institution(s):** Cardiff University


## Problem Background

信息可访问性是基本人权的一部分，但许多公共和官方文档因技术性语言而对部分人群（如非母语者或有认知障碍者）不可理解。
传统的手工文本简化过程资源密集且效率低下，随着信息量增长，依赖人工专家已不可持续，因此需要自然语言处理技术实现自动化文本适应，将复杂西班牙语文本转化为 Plain Language（PL，面向普通大众）和 Easy-to-Read（E2R，面向认知障碍人群）两种格式。

## Method

*   **核心思想:** 利用大型语言模型（LLMs）的生成能力，通过提示工程（Prompt Engineering）引导模型生成符合 PL 和 E2R 标准的简化文本，而无需模型微调。
*   **模型选择:** 初期实验采用 LLaMA-3.2（3B 参数），最终提交使用 Gemma-3（4B 参数），后者在格式一致性和语言准确性上表现更优。
*   **提示策略:** 包括零样本（Zero-shot）、单样本（One-shot）和少样本（Few-shot）提示方法。零样本提示输出不可靠，格式不一致；单样本和少样本提示通过提供简化示例，显著提升输出质量。
*   **提示优化细节:** 通过迭代调整提示内容，解决模型幻觉（Hallucination）、格式错误和事实错误问题。具体措施包括：
    *   要求模型按句子级别处理文本，确保信息保留。
    *   明确输出格式为 Python 字典，便于提取简化结果。
    *   使用目标语言（西班牙语）提示，提升模型在西班牙语文本上的表现。
    *   在提示中加入 E2R 指导方针（如 UNE 153101 EX 标准），包括使用简单词汇、短句、主动语态、避免复杂语法等，确保输出符合目标人群需求。
*   **关键点:** 该方法完全依赖推理阶段的提示设计，不修改模型参数，降低了计算成本，同时通过指导方针确保输出符合特定标准。

## Experiment

*   **有效性:** 最终提交的 Gemma-3 模型（P7 提示）在 CLEARS-2025 共享任务中表现优异，Subtask 1（PL）获得第三名（平均余弦相似度 70%），Subtask 2（E2R）获得第二名（平均余弦相似度 71%）。相比 LLaMA-3.2，Gemma-3 在格式一致性和语言准确性上显著提升，尤其在使用西班牙语提示时。
*   **实验设置:** 基于 CLEARS 语料库（3000 篇西班牙语新闻，70% 训练/30% 测试），评估指标包括 SentenceBERT 余弦相似度、Fernández-Huerta 复杂性评分和 BERTScore (F1)。初步实验在 100-200 句子集上进行，最终提交在完整测试集上评估。
*   **局限性与合理性:** 实验设置较为全面，涵盖不同提示策略和模型对比，但自动评估指标无法捕捉 E2R 文本的视觉格式和可读性需求（如句子分割和排版），模型仍存在幻觉和格式不一致问题，需要后处理。

## Further Thoughts

提示工程无需微调即可显著提升 LLMs 在特定任务上的表现，这种方法可推广至其他语言或领域（如法律、医疗文本简化），通过设计符合目标需求的指导方针引导输出；此外，自动评估指标的局限性启发我们探索结合视觉格式和语义内容的混合评估方法；不同语言提示效果的差异提示 LLMs 的多语言能力可能与文化语境相关，为跨语言文本简化研究提供了新思路。