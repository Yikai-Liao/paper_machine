---
title: "Crosslingual Reasoning through Test-Time Scaling"
pubDatetime: 2025-05-08T16:50:06+00:00
slug: "2025-05-crosslingual-test-scaling"
type: "arxiv"
id: "2505.05408"
score: 0.8859002302471498
author: "grok-3-latest"
authors: ["Zheng-Xin Yong", "M. Farid Adilazuarda", "Jonibek Mansurov", "Ruochen Zhang", "Niklas Muennighoff", "Carsten Eickhoff", "Genta Indra Winata", "Julia Kreutzer", "Stephen H. Bach", "Alham Fikri Aji"]
tags: ["LLM", "Reasoning", "Test Time Scaling", "Multilingual", "Chain of Thought"]
institution: ["Brown University", "MBZUAI", "Stanford University", "University of Tübingen", "Capital One", "Cohere Labs"]
description: "本文通过测试时计算扩展，揭示了英语中心推理模型在多语言数学推理中的潜力，并分析了语言混合模式和跨领域泛化的局限，为多语言推理研究提供了重要基准。"
---

> **Summary:** 本文通过测试时计算扩展，揭示了英语中心推理模型在多语言数学推理中的潜力，并分析了语言混合模式和跨领域泛化的局限，为多语言推理研究提供了重要基准。 

> **Keywords:** LLM, Reasoning, Test Time Scaling, Multilingual, Chain of Thought

**Authors:** Zheng-Xin Yong, M. Farid Adilazuarda, Jonibek Mansurov, Ruochen Zhang, Niklas Muennighoff, Carsten Eickhoff, Genta Indra Winata, Julia Kreutzer, Stephen H. Bach, Alham Fikri Aji

**Institution(s):** Brown University, MBZUAI, Stanford University, University of Tübingen, Capital One, Cohere Labs


## Problem Background

大型语言模型（LLMs）在推理能力上的研究主要集中于英语环境，而多语言预训练模型在英语推理微调后，其跨语言推理能力的泛化性尚未被充分探索。
论文旨在解决如何通过测试时计算扩展（Test-Time Scaling）提升英语中心推理语言模型（RLMs）在多语言任务中的表现，分析语言混合行为、语言强制策略的效果，以及跨领域推理能力的局限性。

## Method

*   **测试时计算扩展（Test-Time Scaling）**：通过增加推理时的计算预算（如增加长链式思考 CoTs 的令牌数量，从 0.5k 到 8k），提升模型在多语言推理任务上的性能，特别是在数学推理领域。实验基于 s1 模型（基于 Qwen2.5-Instruct 的多语言模型，使用 1k 英语 STEM 数据微调），测试不同模型规模（1.5B 到 32B）的效果。
*   **语言混合行为分析**：观察模型在非英语输入下的推理语言选择，发现其以英语为主导，并展现出‘引用与思考’（Quote-and-Think）模式，即引用非英语输入的关键短语并用英语推理，分析其跨语言泛化机制。
*   **语言强制策略**：设计多种方法控制推理语言，包括‘翻译等待’（Translated Wait，通过添加翻译后的‘等待’令牌延长推理）、‘前缀引导’（Prefix，在推理开始时添加目标语言的前缀提示）、‘系统提示’（System Prompt，明确指定推理语言）和‘组合策略’（Combined，综合上述方法），以评估不同语言对推理性能的影响。
*   **跨领域泛化测试**：在 STEM 领域外（如人文、医学、文化常识）测试推理能力的迁移性，分析测试时扩展是否对非训练领域有效。

## Experiment

*   **跨语言推理效果**：在 Multilingual Grade School Math (MGSM) 数据集上，测试时扩展显著提升了 3B 参数以上模型的多语言数学推理性能，例如 s1-14B 模型准确率提升 9.4%（从 0.5k 到 8k 令牌），甚至超越规模两倍的模型（如 R1-Distill-Qwen-32B），尤其在低资源语言上表现优异；但 1.5B 模型提升有限（仅 1.8%），表明模型容量是关键。
*   **语言强制与混合行为**：模型自然以英语为主导推理，‘引用与思考’模式支持跨语言泛化；强制高资源语言（如英语、法语）推理时性能接近或略优于基线，但低资源语言推理性能下降明显，且计算成本高（如斯瓦希里语推理成本是法语的 3.5 倍）。
*   **跨领域泛化**：在 STEM 领域内效果显著（如 Global-MMLU 的 STEM 部分准确率提升明显），但在非 STEM 领域（如医学、人文）增加思考令牌几乎无益，甚至导致性能下降（‘过度思考’问题）。
*   **实验设置评价**：实验覆盖多种模型规模、语言（高资源和低资源）和领域（STEM 和非 STEM），设置较为全面合理；但低资源语言数据较少，跨领域文化特异性分析深度有限，可能影响结论普适性。

## Further Thoughts

测试时计算扩展作为一种无需额外训练即可提升多语言推理性能的方法，启发我们探索动态调整计算预算的可能性；‘引用与思考’模式提示可以通过设计特定提示策略增强跨语言语义理解；高资源语言推理的高效性建议优先使用其作为枢纽语言，但低资源语言推理劣势亟需通过数据增强或多语言微调解决；跨领域泛化局限性则启发我们考虑多领域混合训练或引入文化特异性数据以提升模型适应性。