---
title: "ALLabel: Three-stage Active Learning for LLM-based Entity Recognition using Demonstration Retrieval"
pubDatetime: 2025-09-09T08:47:13+00:00
slug: "2025-09-allabel-active-learning"
type: "arxiv"
id: "2509.07512"
score: 0.6631165368091512
author: "grok-3-latest"
authors: ["Zihan Chen", "Lei Shi", "Weize Wu", "Qiji Zhou", "Yue Zhang"]
tags: ["LLM", "Active Learning", "In-Context Learning", "Entity Recognition", "Sampling"]
institution: ["Beihang University", "Westlake University"]
description: "本文提出 ALLabel 框架，通过三阶段主动学习策略（多样性、相似性、不确定性采样），在有限标注预算下显著提升大语言模型在专业领域实体识别任务中的性能。"
---

> **Summary:** 本文提出 ALLabel 框架，通过三阶段主动学习策略（多样性、相似性、不确定性采样），在有限标注预算下显著提升大语言模型在专业领域实体识别任务中的性能。 

> **Keywords:** LLM, Active Learning, In-Context Learning, Entity Recognition, Sampling

**Authors:** Zihan Chen, Lei Shi, Weize Wu, Qiji Zhou, Yue Zhang

**Institution(s):** Beihang University, Westlake University


## Problem Background

在自然科学领域（如化学和材料科学），命名实体识别（NER）任务需要大规模、高质量的标注数据，而传统微调大语言模型（LLM）方法成本高昂，尤其在标注预算有限的情况下难以实现。
论文旨在解决如何在有限预算下，通过主动学习（Active Learning）挑选最具信息量和代表性的样本进行人工标注，从而构建高效的检索语料库，提升 LLM 在专业领域 NER 任务中的表现。

## Method

*   **核心思想:** 提出 ALLabel 框架，通过三阶段主动学习策略，为 LLM 的上下文学习（In-Context Learning, ICL）选择最具信息量和代表性的演示样本（demonstrations），以降低标注成本并提升实体识别性能。
*   **具体实现:**
    *   **第一阶段 - 多样性采样（Diversity Sampling）:** 基于改进的核心集（Core-Set）算法，使用文本相似性（如 BM25 或 Sentence-BERT）选择初始样本，确保样本代表性；采用‘热启动’（Warm Start）策略，通过选择与其他样本平均相似性最低的样本作为种子数据，避免传统核心集算法的冷启动问题。
    *   **第二阶段 - 相似性采样（Similarity Sampling）:** 基于综合相似性指标（sum_rank），计算每个样本与未标注数据整体的相似性，优先选择与测试输入高度相关的样本，确保 ICL 检索到的演示样本对任务有直接帮助。
    *   **第三阶段 - 不确定性采样（Uncertainty Sampling）:** 利用相似性与不确定性（通过困惑度 perplexity 衡量）的负相关性，挑选 LLM 预测置信度低的样本，针对性地提升模型在‘弱点’上的表现。
*   **关键点:** 三阶段策略按多样性-相似性-不确定性（D-S-U）的顺序执行，比例为 1:3:1，确保逐步构建优化的检索语料库；方法不依赖模型权重更新，仅通过上下文学习实现高效标注。

## Experiment

*   **有效性:** ALLabel 在三个专业领域数据集（CSD-MOFs, NC 2024 General, USPTO）上，显著优于基线方法（如随机采样、传统核心集、困惑度采样），平均 F1 分数提升约 5%-6%。
*   **效率:** 仅标注 5%-10% 的数据，即可达到接近全数据集标注的性能（如 F1 分数差距在 2% 以内），显示出极高的数据效率。
*   **合理性与全面性:** 实验设置涵盖不同数据集、不同 LLM（如 GPT-4o 和 DeepSeek-V3）、不同 shots 数量（1-7），以及其他 NLP 任务（如释义识别和自然语言推理），验证了方法的普适性；消融研究确认了三阶段策略的必要性、最优顺序（D-S-U）和比例（1:3:1）。
*   **局限性:** 实验仅测试了有限的 LLM 模型，未充分考虑文本长度和提取难度对标注预算的影响，未来可进一步扩展验证。

## Further Thoughts

ALLabel 的三阶段主动学习策略（多样性、相似性、不确定性）为数据高效学习提供了新思路，启发我们可以在其他任务中探索多策略结合的样本选择方法；此外，结合领域知识图谱或模型中间层表示来指导采样，可能进一步提升专业领域任务的表现，例如通过语义关系优化多样性采样，而非仅依赖文本相似性。