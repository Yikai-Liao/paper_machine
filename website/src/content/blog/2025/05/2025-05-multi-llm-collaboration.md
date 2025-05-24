---
title: "Collaboration among Multiple Large Language Models for Medical Question Answering"
pubDatetime: 2025-05-22T13:18:45+00:00
slug: "2025-05-multi-llm-collaboration"
type: "arxiv"
id: "2505.16648"
score: 0.7178650024121996
author: "grok-3-latest"
authors: ["Kexin Shang", "Chia-Hsuan Chang", "Christopher C. Yang"]
tags: ["LLM", "Medical QA", "Collaboration", "Reasoning", "Self-Consistency"]
institution: ["Drexel University"]
description: "本文提出多 LLM 迭代协作框架（ICF），通过信息共享和迭代调整显著提升医疗问答任务中的准确率并减少模型分歧，同时探索自信度和一致性对协作效果的影响。"
---

> **Summary:** 本文提出多 LLM 迭代协作框架（ICF），通过信息共享和迭代调整显著提升医疗问答任务中的准确率并减少模型分歧，同时探索自信度和一致性对协作效果的影响。 

> **Keywords:** LLM, Medical QA, Collaboration, Reasoning, Self-Consistency

**Authors:** Kexin Shang, Chia-Hsuan Chang, Christopher C. Yang

**Institution(s):** Drexel University


## Problem Background

大型语言模型（LLMs）在医疗问答（Medical QA）领域展现出潜力，但由于训练数据、架构和调优方式的差异，不同模型在推理能力和表现上存在显著不一致性，尤其在对准确性要求极高的医疗任务中，单一模型的错误或幻觉可能导致严重后果。
作者因此提出研究问题：是否可以通过多个 LLM 的协作，减少模型间的分歧、提升推理能力并改善整体表现？

## Method

*   **核心思想:** 提出一个多 LLM 迭代协作框架（Iterative Collaboration Framework, ICF），通过信息共享和迭代调整，让多个 LLM 在医疗问答任务中减少分歧并提升性能。
*   **具体实现:**
    *   **Zero-shot Chain-of-Thought with Self-Consistency (ZS-CoT-SC):** 初始阶段，每个 LLM 使用零样本链式推理（Chain-of-Thought, CoT）结合自一致性策略，对每个问题生成多个推理路径（重复 10 次），通过多数投票选择最一致的答案，并由外部总结模型（summarizer）提炼推理内容，减少冗余。
    *   **Collaboration Loop:** 针对模型间分歧的问题（即答案不一致），框架进入迭代循环，将所有模型的预测和推理汇总成一个综合文本（transcript），供每个模型重新审视和调整自己的答案；循环持续进行，直到模型间的共识率（consensus rate）达到 80% 或以上。
    *   **模型选择与优化:** 选用三个背景不同的 LLM（Med42、ClinicalCamel、Mixtral）组成团队，确保多样性；通过测试不同提示格式优化模型表现，并将模型量化为 int8 精度以降低计算成本。
*   **关键点:** 该框架不依赖模型间的多轮直接对话，而是通过单向循环和总结机制简化信息流动，同时设置共识率阈值避免过度迭代，确保计算效率。

## Experiment

*   **有效性:** 在 USMLE 数据集（共 305 个问题，涵盖 Step 1、2、3）上，初始 ZS-CoT-SC 阶段模型共识率仅为 50.82%，但经过两轮协作循环后提升至 82.62%，表明协作显著减少了分歧；所有模型准确率均有提升，平均提升幅度为 5.24% 至 6.56%，其中 ClinCamel 提升最明显（6.56%）。
*   **自信度与一致性分析:** 定义了自信度（confidence）作为模型在分歧时坚持原答案的倾向，发现自信度较高的模型（如 Med42，confidence=0.57）初始准确率较高，但协作带来的提升较小；自信度较低的模型（如 ClinCamel，confidence=0.33）从协作中获益更多；一致性（consistency）指标显示模型在正确和错误答案上的表现差异显著，协作后强模型（如 Med42、Mixtral）一致性差距进一步拉大。
*   **实验设置合理性:** 实验涵盖不同难度问题和不同特性模型（医疗专用与通用模型），设置较为全面，但样本量较小（305 个问题），可能影响统计显著性；此外，未深入探讨协作循环的计算成本和扩展性。

## Further Thoughts

1. 自信度与协作收益的关系启发我们可以在多模型系统中设计动态调整机制，根据模型自信度决定是否采纳他人意见，优先让‘弱模型’从协作中获益。
2. 一致性在正确和错误答案上的差异提示可以将一致性作为无监督学习中检测模型幻觉的指标，尤其在缺乏标注数据的场景中具有潜力。
3. ICF 框架的迭代协作理念可扩展至其他复杂任务（如法律推理或决策支持），并可进一步探索角色分工或强化学习机制以优化协作效果。