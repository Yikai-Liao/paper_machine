---
title: "Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO"
pubDatetime: 2025-05-28T15:11:16+00:00
slug: "2025-05-mm-upt-grpo"
type: "arxiv"
id: "2505.22453"
score: 0.7822917568837624
author: "grok-3-latest"
authors: ["Lai Wei", "Yuting Li", "Chen Wang", "Yue Wang", "Linghe Kong", "Weiran Huang", "Lichao Sun"]
tags: ["LLM", "Multi-Modal Learning", "Unsupervised Learning", "Reinforcement Learning", "Reasoning", "Synthetic Data"]
institution: ["Shanghai Jiao Tong University", "Shanghai Innovation Institute", "Zhongguancun Academy", "Lehigh University"]
description: "本文提出 MM-UPT 框架，通过 GRPO 和多数投票机制实现多模态大语言模型的无监督后训练，显著提升推理能力并探索合成数据的可扩展潜力。"
---

> **Summary:** 本文提出 MM-UPT 框架，通过 GRPO 和多数投票机制实现多模态大语言模型的无监督后训练，显著提升推理能力并探索合成数据的可扩展潜力。 

> **Keywords:** LLM, Multi-Modal Learning, Unsupervised Learning, Reinforcement Learning, Reasoning, Synthetic Data

**Authors:** Lai Wei, Yuting Li, Chen Wang, Yue Wang, Linghe Kong, Weiran Huang, Lichao Sun

**Institution(s):** Shanghai Jiao Tong University, Shanghai Innovation Institute, Zhongguancun Academy, Lehigh University


## Problem Background

多模态大语言模型（MLLMs）在后训练阶段通常依赖监督微调（SFT）或强化学习（RL），但这些方法需要大量昂贵的人工标注数据，随着任务复杂性和数量的增加，数据标注变得不可持续。
作者旨在解决这一关键问题：探索在完全无外部监督的情况下，通过无监督后训练方法使 MLLMs 持续自改进其推理能力。

## Method

*   **核心思想:** 提出 MM-UPT（Multi-Modal Unsupervised Post-Training）框架，利用在线强化学习算法 GRPO，通过自奖励机制实现无监督自改进，避免对外部标注数据或奖励模型的依赖。
*   **具体实现:** 
    *   基于 GRPO 算法，MM-UPT 对无标签的多模态输入（如图像和问题）采样生成多个响应。
    *   使用多数投票（majority voting）从多个响应中选出最一致的答案作为伪标签（pseudo-label），并基于是否与伪标签一致计算每个响应的奖励信号。
    *   通过 GRPO 的优势估计（advantage estimation）和策略更新机制，优化模型以倾向于生成更一致、高共识的响应。
    *   引入 KL 散度约束，限制模型偏离参考策略，确保训练稳定性。
*   **数据场景:** 除了使用人类创建的无标签问题，MM-UPT 还探索了模型自生成的合成数据，包括上下文合成（In-Context Synthesizing，利用图像、问题和答案生成新问题）和直接合成（Direct Synthesizing，仅基于图像生成问题），以进一步减少对外部数据的依赖。
*   **关键创新:** 不需要外部监督信号，依靠模型自身的响应一致性驱动改进，同时保持计算效率（无需单独的价值模型）。

## Experiment

*   **有效性:** MM-UPT 在多个多模态数学推理基准数据集上显著提升了模型性能，例如在 Qwen2.5-VL-7B 模型上，MathVista 准确率从 66.3% 提升至 72.9%，We-Math 从 62.9% 提升至 68.7%，平均提升约 3-7%。
*   **对比优势:** 相较于其他无监督基线方法（如 LMSI, SRLM, Genixer, STIC），MM-UPT 表现更优，甚至接近监督方法（如 GRPO 和 SFT）的效果，显示出无监督自改进的潜力。
*   **实验设置合理性:** 实验覆盖了两种无标签数据场景：人类创建的问题（无答案）和模型自生成的合成问题。合成数据实验表明，直接合成和上下文合成均能带来显著提升，验证了自生成数据的可行性。
*   **局限性分析:** 在模型对数据集缺乏足够先验知识时（如 ThinkLite-11K 数据集），多数投票可能放大错误，导致性能下降（平均准确率从 49.47% 降至 44.11%），这一分析揭示了方法的适用边界。
*   **计算开销:** 主要开销在于采样多个响应和计算多数投票奖励，但整体基于 GRPO 的高效设计，未引入额外复杂模型，保持了训练的可行性。

## Further Thoughts

多数投票作为伪奖励信号的机制启发了我，可以在其他无监督学习任务中探索类似的共识驱动策略，以提升模型一致性和稳定性；此外，合成数据的成功应用让我思考是否可以通过更复杂的生成策略（如多模型协作或对抗生成）进一步提升数据质量；同时，MM-UPT 在缺乏先验知识时的失败案例提示是否可以设计动态奖励机制，根据任务难度或模型置信度切换奖励策略（如从多数投票切换到自评估或外部协作验证）。