---
title: "Joint Flashback Adaptation for Forgetting-Resistant Instruction Tuning"
pubDatetime: 2025-05-21T12:45:28+00:00
slug: "2025-05-flashback-adaptation"
type: "arxiv"
id: "2505.15467"
score: 0.8609414090718526
author: "grok-3-latest"
authors: ["Yukun Zhao", "Lingyong Yan", "Zhenyang Li", "Shuaiqiang Wang", "Zhumin Chen", "Zhaochun Ren", "Dawei Yin"]
tags: ["LLM", "Continual Learning", "Instruction Tuning", "Catastrophic Forgetting", "Knowledge Sharing"]
institution: ["Baidu Inc., Beijing, China", "Shandong University, Jinan, China", "Leiden University, Leiden, The Netherlands"]
description: "本文提出联合闪回适应方法，通过少量旧任务提示和潜在任务插值，实现大型语言模型在新任务适应中有效缓解灾难性遗忘。"
---

> **Summary:** 本文提出联合闪回适应方法，通过少量旧任务提示和潜在任务插值，实现大型语言模型在新任务适应中有效缓解灾难性遗忘。 

> **Keywords:** LLM, Continual Learning, Instruction Tuning, Catastrophic Forgetting, Knowledge Sharing

**Authors:** Yukun Zhao, Lingyong Yan, Zhenyang Li, Shuaiqiang Wang, Zhumin Chen, Zhaochun Ren, Dawei Yin

**Institution(s):** Baidu Inc., Beijing, China, Shandong University, Jinan, China, Leiden University, Leiden, The Netherlands


## Problem Background

大型语言模型（LLMs）在现实应用中需要通过增量任务持续微调以适应用户指令和偏好，但面临灾难性遗忘（Catastrophic Forgetting）问题，即学习新任务时会丢失旧任务知识。
现有方法如经验回放、优化约束和任务区分存在局限性（如数据隐私问题、高计算成本、需任务标识），因此需要一种更实用、泛化的方法，在不依赖回放数据或任务区分的情况下，同时提升新任务泛化能力和旧任务性能保留。

## Method

*   **核心思想:** 提出联合闪回适应（Joint Flashback Adaptation, JFA），通过少量旧任务提示（Flashbacks）作为锚点约束模型输出偏差，并结合潜在任务插值促进新旧任务知识共享，实现任务无关的持续学习。
*   **闪回机制:** 从旧任务中选取少量提示（可手动编写或从验证集采样），计算模型输出与原始模型在这些提示上的双向KL散度损失（Divergence Loss），以此约束模型避免遗忘旧任务知识。
*   **联合任务学习:** 引入潜在任务（Latent Tasks）作为新旧任务之间的插值，潜在任务以键值对形式表示（键为任务编码向量，值为核心知识权重增量），通过输入编码的余弦相似性检索相关潜在任务，并结合LoRA技术进行参数高效更新，缓解闪回数据稀疏性，促进知识共享。
*   **梯度投影优化:** 采用PCGrad方法，通过梯度投影缓解新任务学习和旧任务保留之间的冲突，确保稳定-可塑性平衡。
*   **特点:** 方法仅需少量闪回提示，无需回放完整数据或任务区分，适用于资源受限场景。

## Experiment

*   **有效性:** 实验在 Vicuna-13B 和 Llama3.1-8B 模型上进行，覆盖1000+指令跟随、算术推理和通用推理任务（数据集包括 Super Natural Instructions, GSM-8K, SVAMP, ARC-Challenge, MMLU-Pro, BBH），结果显示 JFA 在新任务上的泛化性能（BLEU, ROUGE 指标）优于基线方法（如 SFT, PACE, Replay, FROMP, SLM），在旧任务上的准确率表现最佳或次佳，表明其在缓解灾难性遗忘方面效果显著。
*   **对比分析:** 与直接微调（SFT）相比，JFA 避免了旧任务性能的显著下降；与需要回放数据的 Replay 和 FROMP 相比，JFA 仅用少量闪回提示（每任务30个）就达到了类似甚至更好的效果，资源效率更高。
*   **实验设置合理性:** 实验覆盖多种任务类型和数据集，评估指标全面（新任务用 BLEU/ROUGE，旧任务用准确率），基线选择涵盖主流持续学习方法，设置较为合理；同时对闪回数量、权重参数和联合任务学习超参数进行了详细消融分析，验证了各组件的有效性。
*   **不足与开销:** 实验主要在已对齐的强模型上验证，缺乏对弱模型的测试，可能影响方法普适性；训练效率略低于简单 SFT，但与 SLM 相当，GPU 内存消耗与基线接近。

## Further Thoughts

闪回机制的创新在于用少量提示作为锚点即可有效缓解遗忘，启发我们在资源受限场景下通过设计代表性提示保护模型核心能力，而无需存储大量历史数据；联合任务学习通过潜在任务插值促进知识共享，这一思路可扩展至多模态或跨领域持续学习，探索更精细的任务共性捕捉方法；此外，梯度投影在平衡新旧任务冲突中的作用值得深入研究，或许可以结合动态权重调整或更精细的梯度分析技术，进一步优化持续学习效果。