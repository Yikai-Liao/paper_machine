---
title: "Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning"
pubDatetime: 2025-05-14T19:00:27+00:00
slug: "2025-05-tokenizer-flexibility-adaptation"
type: "arxiv"
id: "2505.09738"
score: 0.6764075314047862
author: "grok-3-latest"
authors: ["Shaurya Sharthak", "Vinayak Pahalwan", "Adithya Kamath", "Adarsh Shirawalmath"]
tags: ["LLM", "Tokenizer Adaptation", "Embedding Initialization", "Compression Efficiency", "Multilingual Processing"]
institution: ["Tinycompany", "Proton", "Tensoic", "Google TRC"]
description: "本文提出 TokenAdapt 框架和 Supertoken 学习，通过混合启发式初始化和概率预分词策略，实现大型语言模型分词器的低成本灵活替换和压缩效率提升，显著改善零-shot 性能和多领域适应性。"
---

> **Summary:** 本文提出 TokenAdapt 框架和 Supertoken 学习，通过混合启发式初始化和概率预分词策略，实现大型语言模型分词器的低成本灵活替换和压缩效率提升，显著改善零-shot 性能和多领域适应性。 

> **Keywords:** LLM, Tokenizer Adaptation, Embedding Initialization, Compression Efficiency, Multilingual Processing

**Authors:** Shaurya Sharthak, Vinayak Pahalwan, Adithya Kamath, Adarsh Shirawalmath

**Institution(s):** Tinycompany, Proton, Tensoic, Google TRC


## Problem Background

大型语言模型（LLMs）在预训练阶段与特定分词器（tokenizer）紧密耦合，导致‘tokenizer lock-in’问题，表现为在多语言或特定领域任务中词汇表与目标数据不匹配，引发语义失真和处理效率低下（如 token 碎片化增加计算成本和推理延迟）；传统解决方案（如词汇扩展结合持续预训练 CPT）成本高昂，且无法根本解决原始分词器合并策略的低效问题，因此需要在不牺牲预训练知识的前提下，以低成本实现分词器灵活替换并提升效率。

## Method

* **TokenAdapt 框架**：一个模型无关的分词器移植方法，核心是通过‘混合启发式’（hybrid heuristic）为新词汇表中的独特 token 初始化嵌入，以保留语义关系并减少后续训练需求：
  * **局部启发式（Local Heuristic）**：利用原始分词器将新 token 分解为子 token，通过外部文本嵌入模型计算新 token 与子 token 的语义相似度，并结合长度归一化（length normalization）加权，合成局部估计的嵌入，确保新 token 的语义与原始子 token 结构相关。
  * **全局启发式（Global Heuristic）**：在外部嵌入空间中，通过 k 近邻（kNN）搜索找到原始词汇表中与新 token 语义最相似的 token，按相似度加权合成全局估计的嵌入，捕捉更广泛的语义上下文。
  * **混合整合**：通过超参数加权组合局部和全局估计，形成最终嵌入，旨在零-shot 条件下最大化语义一致性，同时支持 tied 和 untied 嵌入配置的 Transformer 架构。
* **Supertoken 学习**：通过概率预分词策略（probabilistic pre-tokenization），在 BPE 训练前对文本进行随机分块并插入分隔符，鼓励学习跨词的多词 token（multi-word tokens），从而提升序列压缩效率，减少 token 碎片化，尤其适用于多语言和专业领域文本。

## Experiment

* **TokenAdapt 有效性**：在零-shot 困惑度比（perplexity ratio）评估中，TokenAdapt 的混合启发式方法显著优于基线（如 ReTok 和 TransTokenizer），例如在 Llama-3.2-3B 到 QTK-81K 移植中，整体困惑度比为 48.2，相比 ReTok 的 71.1 和 TransTokenizer 的 145.9，改进约 2 倍，显示出更好的性能保留。
* **领域适应性**：在英语、印地语、代码、数学等多个领域中，TokenAdapt 表现均衡，尤其在印地语和代码领域困惑度比大幅降低，表明其对多语言和专业领域的适应能力。
* **Supertoken 压缩效果**：Adi-Bun-128K 分词器在多领域（如英语、印地语、代码）中 token 总数显著低于 DeepSeek-R1、Krutrim-Ins 和 Gemma-3-27b，验证了 supertoken 减少碎片化的潜力。
* **实验设置合理性与局限**：实验覆盖多种基础模型（Llama-3.2-3B、Qwen2.5-3B）、目标分词器和领域，数据集（tinycompany/ppl）包含多语言和多领域子集，零-shot 评估直接反映初始化质量，设计全面；但未提供后续微调性能数据，限制了对长期效果的评估；此外，相似度阈值策略的失败实验提供了嵌入空间复杂性的洞察。

## Further Thoughts

TokenAdapt 的混合启发式方法揭示了嵌入空间中局部结构与全局语义的互补性，启发在其他模型迁移任务中探索多层次语义映射策略；Supertoken 的概率预分词策略提示传统子词分词的局限，跨词 token 或动态分词可能进一步提升效率，尤其在资源受限场景下；相似度阈值策略的失败表明嵌入空间的非线性特性，未来可尝试非线性映射方法优化初始化。