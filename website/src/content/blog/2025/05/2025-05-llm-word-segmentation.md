---
title: "Segment First or Comprehend First? Explore the Limit of Unsupervised Word Segmentation with Large Language Models"
pubDatetime: 2025-05-26T07:48:15+00:00
slug: "2025-05-llm-word-segmentation"
type: "arxiv"
id: "2505.19631"
score: 0.7666118830581332
author: "grok-3-latest"
authors: ["Zihong Zhang", "Liqi He", "Zuchao Li", "Lefei Zhang", "Hai Zhao", "Bo Du"]
tags: ["LLM", "Unsupervised Learning", "Word Segmentation", "Semantic Understanding"]
institution: ["Wuhan University", "Shanghai Jiao Tong University"]
description: "本文提出‘先理解，后分词’范式，利用大型语言模型（LLMs）的语义能力显著提升无监督分词效果，并通过 LLACA 方法结合 Aho-Corasick 自动机实现高效鲁棒的分词工具，重新定义了无监督分词的上限。"
---

> **Summary:** 本文提出‘先理解，后分词’范式，利用大型语言模型（LLMs）的语义能力显著提升无监督分词效果，并通过 LLACA 方法结合 Aho-Corasick 自动机实现高效鲁棒的分词工具，重新定义了无监督分词的上限。 

> **Keywords:** LLM, Unsupervised Learning, Word Segmentation, Semantic Understanding

**Authors:** Zihong Zhang, Liqi He, Zuchao Li, Lefei Zhang, Hai Zhao, Bo Du

**Institution(s):** Wuhan University, Shanghai Jiao Tong University


## Problem Background

论文探讨了大型语言模型（LLMs）在无监督分词任务中的潜力，提出‘先理解，后分词’的新范式，挑战传统‘先分词，后理解’的 NLP 流程。
核心问题在于如何利用 LLMs 的语义理解能力提升无监督分词效果，尤其是在中文、日文等无明确词边界的语言中，同时通过分词任务评估 LLMs 的细粒度语言理解能力，解决传统方法在歧义处理和领域适应性上的不足。

## Method

*   **LLM-WS（LLM Word Segmentation）**：
    *   直接利用 LLMs 的语义理解能力，通过简单提示（prompt）指导模型对原始文本进行分词。
    *   研究测试了不同参数规模的 LLMs（如 Qwen1.5 系列从 7B 到 72B），观察到参数规模与分词效果的正相关性。
    *   方法优点在于无需标注数据，依赖模型预训练的语义知识，但推理成本高。
*   **LLACA（Large Language Model-Inspired Aho-Corasick Automaton）**：
    *   创新性地结合 LLMs 的语义理解与 Aho-Corasick 自动机的高效模式匹配能力，提出一种无监督分词框架。
    *   具体步骤包括：
        1. 使用 LLMs 对批量文本进行初步分词，生成候选词并计算词频。
        2. 应用点互信息（PMI）过滤不合理的词，确保候选词的语义一致性。
        3. 将过滤后的词集成到 Aho-Corasick 自动机中，形成动态词典。
        4. 引入变长 n-gram 模型计算词概率，结合 Viterbi 解码算法寻找最优分词路径。
    *   优点在于既保留了 LLMs 的深层语义理解，又通过自动机实现高效推理，同时对噪声和未登录词（OOV）具有鲁棒性。
    *   与传统方法相比，LLACA 动态适应语言变化，无需人工标注词典。

## Experiment

*   **LLM-WS 效果**：在中文数据集（如 MSR、PKU）上，Qwen1.5-72B 的 F-measure 达到 88.7，显著优于传统无监督方法（约 80），接近监督方法水平；但在非主要预训练语言（如泰文）上表现较差，F-measure 仅为 21.3，反映模型对训练语料的依赖性。
*   **LLACA 效果**：LLACA 在保持高准确率的同时大幅提升推理效率，例如在 MSR 数据集上，Qwen1.5-14B-Chat 推理耗时 3.65 小时（F-measure 85.9），而 LLACA 仅需 2.01 秒（F-measure 提升至 87.7）；在跨领域和 OOV 处理上，LLACA 也优于传统方法，如在 MSR 数据集的 OOV 测试中 F-measure 达 86.7，远超 SLM-3 的 73.9。
*   **实验设置**：实验覆盖中文、日文、韩文、泰文等多种语言数据集（如 SIGHAN Bakeoff 2005、KWDLC），评估指标为 F-measure，设置较为全面；但对非中文语言的表现分析较少，且未探讨提示设计对结果的影响，可能存在一定局限性。

## Further Thoughts

‘先理解，后分词’的范式转变启发我们重新思考 NLP 任务流程，是否可以在更多任务（如句法分析）中以语义理解为起点，而非依赖预处理；LLACA 的混合设计（LLMs + 传统算法）提示了一种高效工具开发思路，是否可推广至其他领域如命名实体识别；此外，分词作为语义理解的评估工具这一观点，是否意味着我们可以设计更多细粒度任务来测试 LLMs 的语言能力层次？