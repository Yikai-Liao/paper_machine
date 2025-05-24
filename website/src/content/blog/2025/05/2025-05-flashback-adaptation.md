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
description: "本文提出联合闪回适应方法，通过少量旧任务提示和联合任务学习，实现大型语言模型在指令微调中的持续学习，有效缓解灾难性遗忘问题。"
---

> **Summary:** 本文提出联合闪回适应方法，通过少量旧任务提示和联合任务学习，实现大型语言模型在指令微调中的持续学习，有效缓解灾难性遗忘问题。 

> **Keywords:** LLM, Continual Learning, Instruction Tuning, Catastrophic Forgetting, Knowledge Sharing

**Authors:** Yukun Zhao, Lingyong Yan, Zhenyang Li, Shuaiqiang Wang, Zhumin Chen, Zhaochun Ren, Dawei Yin

**Institution(s):** Baidu Inc., Beijing, China, Shandong University, Jinan, China, Leiden University, Leiden, The Netherlands


## Problem Background

大型语言模型（LLMs）在实际应用中需要持续微调以适应新任务或用户指令，但面临灾难性遗忘（Catastrophic Forgetting）问题，即学习新任务时会遗忘旧任务的知识。
现有方法如经验回放、优化约束和任务区分在现实场景中受限于数据隐私、高计算成本或需要任务标识等问题，论文旨在提出一种更实用、任务无关的方法，既能在新任务上泛化良好，又能保留旧任务能力。

## Method

*   **核心思想：** 提出联合闪回适应（Joint Flashback Adaptation, JFA），通过少量旧任务提示（Flashbacks）和联合任务学习（Joint Task Learning）缓解灾难性遗忘，同时保持新任务学习能力。
*   **闪回机制：** 从旧任务中选取少量提示作为锚点数据（可手动编写或从验证集采样），通过发散损失（Divergence Loss）限制当前模型输出与原始模型输出的偏差，避免遗忘旧知识，且无需访问完整训练数据或标签。
*   **联合任务学习：** 引入潜在任务（Latent Tasks）作为新任务和闪回之间的插值，潜在任务以键值对形式表示（键为向量编码，值为权重增量），通过输入编码的相似性检索（KNN）找到相关潜在任务，结合权重增量更新模型参数，缓解闪回数据稀疏性并促进知识共享。
*   **梯度投影：** 使用 PCGrad 方法协调新任务学习和旧任务保留之间的梯度冲突，确保更新方向不会相互干扰。
*   **特点：** 方法任务无关（Task-Agnostic），不依赖任务类型或标识，仅需少量闪回提示，适用于资源受限场景。

## Experiment

*   **有效性：** 在 Vicuna-13B 和 Llama3.1-8B 模型上，JFA 在新任务（Super Natural Instructions）上的泛化性能（BLEU, ROUGE 等指标）显著优于基线方法（如 SFT, PACE, Replay, FROMP, SLM），提升明显；同时在旧任务（GSM-8K, SVAMP, ARC-Challenge, MMLU-Pro, BIG-Bench-Hard）上的准确率表现最佳或接近最佳，表明有效缓解了灾难性遗忘。
*   **对比分析：** 与直接微调（SFT）相比，JFA 避免了旧任务性能的显著下降；与需要大量回放数据的 Replay 和 FROMP 相比，JFA 仅用少量闪回提示即可达到类似效果，资源效率更高。
*   **实验设置合理性：** 实验覆盖了1000+指令跟随、算术推理和通用推理任务，测试了多种数据集和评估指标，设置全面；通过消融实验分析了闪回数量、联合任务学习参数（C, Q, k）和权重因子 α 的影响，验证了方法的鲁棒性。
*   **不足与局限：** 论文未在较弱模型上验证效果，且未细粒度区分模型已有能力和新任务需求，可能影响适用范围。

## Further Thoughts

闪回机制启发我们可以在资源受限场景下通过少量代表性数据作为锚点维持模型性能，未来可探索如何自动生成高质量闪回提示；
联合任务学习通过潜在任务插值实现知识共享，这一思路可扩展到多模态学习或跨领域适应，通过构建中间表示缓解数据稀疏问题；
梯度投影（PCGrad）在多目标优化中的应用值得关注，未来可研究更复杂的梯度协调策略，以进一步平衡新旧任务的学习冲突。