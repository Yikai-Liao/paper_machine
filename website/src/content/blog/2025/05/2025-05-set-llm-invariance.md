---
title: "Set-LLM: A Permutation-Invariant LLM"
pubDatetime: 2025-05-21T12:14:26+00:00
slug: "2025-05-set-llm-invariance"
type: "arxiv"
id: "2505.15433"
score: 0.5838568907290309
author: "grok-3-latest"
authors: ["Beni Egressy", "Jan Stühmer"]
tags: ["LLM", "Permutation Invariance", "Attention Mechanism", "Positional Encoding", "Evaluation"]
institution: ["Heidelberg Institute for Theoretical Studies", "Karlsruhe Institute of Technology"]
description: "本文提出 Set-LLM，一种首个针对解码器型大型语言模型的顺序不变架构，通过定制化的位置编码和注意力掩码消除顺序敏感性，同时保持性能和运行效率。"
---

> **Summary:** 本文提出 Set-LLM，一种首个针对解码器型大型语言模型的顺序不变架构，通过定制化的位置编码和注意力掩码消除顺序敏感性，同时保持性能和运行效率。 

> **Keywords:** LLM, Permutation Invariance, Attention Mechanism, Positional Encoding, Evaluation

**Authors:** Beni Egressy, Jan Stühmer

**Institution(s):** Heidelberg Institute for Theoretical Studies, Karlsruhe Institute of Technology


## Problem Background

大型语言模型（LLMs）在众多应用中表现出色，但其鲁棒性仍是一个关键问题，尤其体现在对输入顺序的敏感性（Order Sensitivity）。
这种敏感性表现为模型在处理多选题或选项排序任务时，因选项顺序变化而给出不同答案，甚至导致性能显著下降，特别是在用作自动化评估器（LLM-as-a-judge）时，顺序偏见直接影响评估可靠性。
论文的出发点是消除这种顺序敏感性，确保模型在面对输入顺序变化时具有不变性（Permutation Invariance），从而提高其在高风险领域和自动化评估中的可靠性。

## Method

*   **核心思想:** 通过对预训练大型语言模型（LLMs）进行架构适配，构建一种顺序不变的模型（Set-LLM），使其对混合集合-文本输入的顺序变化具有鲁棒性，同时不牺牲性能或增加计算复杂度。
*   **具体实现步骤:**
    *   **移除顺序依赖组件:** 首先移除传统的顺序位置编码（Positional Encoding, PE）和因果注意力掩码（Causal Mask），将模型转变为‘词袋’模型（Bag-of-Words, BoW），从而忽略输入中所有 token 的顺序信息。
    *   **引入集合位置编码（SetPE）:** 设计一种新的位置编码方式，为集合内的每个元素分配相同的起始位置编码，确保集合内元素顺序无关，但保留元素内部的 token 顺序信息（例如，多选题选项内部的词序）。
    *   **引入集合注意力掩码（SetMask）:** 基于前缀掩码（Prefix Mask），进一步限制集合内不同元素之间的注意力交互（即不同选项之间不直接交互），确保模型能区分集合内的不同元素，避免混淆。
    *   **理论保证:** 通过数学证明，Set-LLM 在架构上实现了集合顺序不变性（Set Permutation Invariance），即对集合内元素的排列顺序变化，模型输出保持一致。
*   **关键特点:** 该方法不依赖于额外训练数据或后处理（如多数投票），而是直接通过架构设计实现不变性；适配过程可应用于不同的解码器型 LLMs，且不依赖特定模型版本。

## Experiment

*   **有效性:** Set-LLM 在四个多选题数据集（PIQA, ARC-Challenge, CommonsenseQA, SIQA）上测试，显示出显著的顺序不变性；在随机顺序（Random Order）和对抗顺序（Adversarial Order）两种评估模式下，准确率无任何下降（例如在 ARC-Challenge 上保持 65.02%），而基线模型（如 Gemma 2B）在对抗顺序下准确率大幅下降（从 55.20% 降至 23.72%）。
*   **性能提升:** Set-LLM 在 20/20 的对抗顺序场景和 18/20 的随机顺序场景中优于基线模型，且单次运行即可达到甚至超过基线模型的多数投票（Majority Vote）结果，避免了指数级的运行开销。
*   **实验设置合理性:** 实验覆盖了多种模型架构（Gemma 2B/7B, Llama 3.2 1B/3B, Llama 3.1 8B）和数据集，测试了不同规模和类型的模型；此外，还测试了分布外（Out-of-Distribution）性能，Set-LLM 在 10/12 的场景中表现最佳，表明其泛化能力较强。
*   **计算开销:** Set-LLM 的运行时间和内存使用与基线模型相当（例如 Gemma 2B 上的评估时间仅从 357.63 秒增加到 365.47 秒），证明其未增加显著计算负担。

## Further Thoughts

Set-LLM 通过架构设计直接嵌入顺序不变性（Permutation Invariance）的思路非常具有启发性，是否可以将其他类型的 invariance（如对输入格式、噪声的鲁棒性）也通过类似的方式嵌入模型架构？例如，设计对多模态输入顺序无关的注意力机制，或将 Set-LLM 的集合处理方式应用于检索增强生成（RAG）中对支持文档集合的无序处理，甚至结合图神经网络（GNN）处理更复杂的结构化数据。