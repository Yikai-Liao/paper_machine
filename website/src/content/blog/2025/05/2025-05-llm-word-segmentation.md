---
title: "Segment First or Comprehend First? Explore the Limit of Unsupervised Word Segmentation with Large Language Models"
pubDatetime: 2025-05-26T07:48:15+00:00
slug: "2025-05-llm-word-segmentation"
type: "arxiv"
id: "2505.19631"
score: 0.7666118830581332
author: "grok-3-latest"
authors: ["Zihong Zhang", "Liqi He", "Zuchao Li", "Lefei Zhang", "Hai Zhao", "Bo Du"]
tags: ["LLM", "Word Segmentation", "Unsupervised Learning", "Context Understanding", "Pattern Matching"]
institution: ["Wuhan University", "Shanghai Jiao Tong University"]
description: "本文提出基于大型语言模型的无监督词分割框架 LLM-WS 和 LLACA，通过‘先理解，后分割’的理念显著提升多语言词分割性能，并为评估 LLMs 语义理解能力提供了新视角。"
---

> **Summary:** 本文提出基于大型语言模型的无监督词分割框架 LLM-WS 和 LLACA，通过‘先理解，后分割’的理念显著提升多语言词分割性能，并为评估 LLMs 语义理解能力提供了新视角。 

> **Keywords:** LLM, Word Segmentation, Unsupervised Learning, Context Understanding, Pattern Matching

**Authors:** Zihong Zhang, Liqi He, Zuchao Li, Lefei Zhang, Hai Zhao, Bo Du

**Institution(s):** Wuhan University, Shanghai Jiao Tong University


## Problem Background

词分割是许多语言（如中文、日文）中自然语言处理（NLP）的关键基础步骤，因这些语言缺乏明确的词边界，传统方法多遵循‘先分割，后理解’的范式，依赖统计模型或人工标注，难以处理歧义和领域适应性问题。
本文提出‘先理解，后分割’的新视角，探索大型语言模型（LLMs）在无监督词分割中的潜力，并通过该任务评估 LLMs 的语义理解能力，旨在解决传统方法的局限性。

## Method

*   **核心框架 LLM-WS（LLM Word Segmentation）**：
    *   利用 LLMs 的强大语义理解能力，通过简单提示（Prompt）直接对原始文本进行词分割。
    *   模型根据上下文理解动态分割词语，无需预先标注数据，特别适用于处理歧义句子。
*   **创新方法 LLACA（Large Language Model-Inspired Aho-Corasick Automaton）**：
    *   结合 LLMs 的语义洞察与 Aho-Corasick 自动机（AC 自动机）的高效模式匹配能力，构建动态 n-gram 模型。
    *   具体步骤：首先通过 LLM-WS 从文本中提取分割模式并计算词频；然后使用点互信息（PMI）过滤噪声数据；接着将模式整合到 AC 自动机中，基于上下文调整概率分布；最后通过 Viterbi 解码寻找最优分割路径。
    *   创新点在于动态构建词汇表，结合上下文信息解决歧义问题，同时利用 AC 自动机保证高效性。
*   **优势与特点**：
    *   无监督性：无需人工标注，降低数据获取成本。
    *   适应性：通过 LLMs 动态适应语言变化和未登录词（OOV）。
    *   高效性：LLACA 将 LLMs 的语义能力转化为轻量级工具，显著降低推理成本。

## Experiment

*   **LLM-WS 效果**：在中文数据集（如 MSR、PKU）上，LLMs（如 Qwen1.5-72B）表现突出，F-measure 达到 88.2，显著优于传统无监督方法的约 80；但在非主要训练语言（如日文 KWDLC、泰文 BEST）上易产生幻觉，性能较低（例如 KWDLC 为 70.3）。
*   **LLACA 提升**：LLACA 在 LLM-WS 基础上进一步优化，消除幻觉并利用上下文信息，在多语言任务中均有提升（例如 MSR 从 85.9 提升到 87.7，KWDLC 从 70.3 提升到 76.7）；与传统工具（如 Jieba）相比，推理速度从小时级降至秒级（例如 MSR 数据集从 3.65 小时降至 2.01 秒）。
*   **实验设置合理性**：实验覆盖中文、日文、韩文、泰文等多种语言数据集（如 SIGHAN Bakeoff 2005 的 MSR、PKU，泰文的 BEST），具有代表性；通过 Qwen1.5 系列（7B 到 72B）验证了参数量与性能的正相关性。
*   **不足之处**：对小规模模型局限性和非主导语言幻觉问题的解决方案探讨不足。

## Further Thoughts

‘先理解，后分割’的理念启发我们重新思考 NLP 任务的范式，LLMs 的端到端语义理解能力可能不仅限于词分割，还能扩展到句法分析、语义角色标注等任务，减少对中间步骤的依赖；此外，LLACA 的设计提示我们可以通过‘蒸馏’ LLMs 的语义能力到轻量级工具中，解决推理成本高的问题；另一个方向是探索多模态 LLMs（结合图像、语音）是否能进一步提升上下文理解能力，尤其在处理口语或多语言混合文本时。