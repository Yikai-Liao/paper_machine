---
title: "Can Large Language Models Develop Strategic Reasoning? Post-training Insights from Learning Chess"
pubDatetime: 2025-07-01T13:16:34+00:00
slug: "2025-07-strategic-reasoning-chess"
type: "arxiv"
id: "2507.00726"
score: 0.7500257377823053
author: "grok-3-latest"
authors: ["Dongyoon Hwang", "Hojoon Lee", "Jaegul Choo", "Dongmin Park", "Jongho Park"]
tags: ["LLM", "Strategic Reasoning", "Reinforcement Learning", "Knowledge Distillation", "Chess"]
institution: ["KAIST", "KRAFTON", "UC Berkeley"]
description: "本文通过象棋任务探索了大型语言模型在战略推理领域的潜力，提出基于密集奖励的强化学习方法，并揭示预训练领域知识对 RL 效果的关键影响。"
---

> **Summary:** 本文通过象棋任务探索了大型语言模型在战略推理领域的潜力，提出基于密集奖励的强化学习方法，并揭示预训练领域知识对 RL 效果的关键影响。 

> **Keywords:** LLM, Strategic Reasoning, Reinforcement Learning, Knowledge Distillation, Chess

**Authors:** Dongyoon Hwang, Hojoon Lee, Jaegul Choo, Dongmin Park, Jongho Park

**Institution(s):** KAIST, KRAFTON, UC Berkeley


## Problem Background

大型语言模型（LLMs）在数学推理等逻辑任务上通过强化学习（RL）取得了显著进展，但其在战略推理（Strategic Reasoning）领域的潜力仍未被充分探索。
战略推理涉及在多智能体环境中进行规划、预测对手行为并做出决策，更贴近现实世界的复杂场景。
本文以国际象棋（Chess）为测试平台，研究 LLMs 是否能通过强化学习发展出战略推理能力，解决现有研究中关于战略推理能力的空白问题。

## Method

*   **核心思想:** 通过强化学习的可验证奖励（RLVR）对 LLMs 进行后训练（Post-Training），以培养其在象棋任务中的战略推理能力，探索不同奖励机制对学习效果的影响。
*   **具体实现:** 
    *   **训练框架:** 采用 Group Relative Policy Optimization (GRPO) 算法对 Qwen2.5 和 Llama3.1 模型进行训练，使用 Lichess 象棋谜题数据集（19.2k 样本），将谜题分解为位置-动作对。
    *   **奖励机制:** 设计两种奖励方式：
        - **稀疏奖励（Sparse Reward）:** 仅基于预测移动是否正确（二元反馈），即是否与最优移动匹配。
        - **密集奖励（Dense Reward）:** 利用预训练的象棋专家模型（Action-Value Network）提供连续反馈，基于移动后的胜率预测，视为一种知识蒸馏（Knowledge Distillation），不仅评估最优移动，还对次优移动提供分级反馈。
        - 附加辅助奖励，确保输出格式正确（如使用 <think> 和 <answer> 标签）和语言为英语。
    *   **输入设计:** 提示采用 Forsyth-Edwards Notation (FEN) 表示棋盘状态，Standard Algebraic Notation (SAN) 表示移动，模型需输出推理过程和最终答案。
    *   **额外尝试:** 使用 OpenAI o3 模型生成的推理轨迹进行监督微调（SFT），以增强模型推理能力，随后再进行 RL 训练。
*   **关键点:** 密集奖励通过专家模型的指导提供更细粒度的反馈，旨在帮助模型学习复杂的战略推理，而不仅仅是动作预测。

## Experiment

*   **有效性:** 密集奖励显著优于稀疏奖励，尤其在 Qwen2.5-3B 和 Llama3.1-8B 模型上，稀疏奖励几乎完全失效；密集奖励模型也优于监督微调（SFT）基线，表明专家指导的反馈对推理能力提升有帮助。
*   **局限性:** 尽管有改进，所有模型的象棋谜题准确率在 25-30% 左右达到平台期，远低于人类专家水平（60-80%）或 1800 ELO 模型（66.5%）。
*   **推理 SFT 效果:** 使用 OpenAI o3 推理轨迹进行 SFT 后，模型输出更结构化，但 RL 训练后的准确率未显著提升，甚至在 Llama3.1-8B 上有所下降。
*   **失败分析:** 通过诊断任务发现，模型对基本象棋规则的内部理解不足，无法准确跟踪游戏状态或识别基本战术，这是性能瓶颈的主要原因。
*   **实验设置合理性:** 实验涵盖不同模型规模（3B、7B、8B）、奖励机制（稀疏 vs 密集）和训练方式（SFT+RL），并通过消融研究验证了提示格式和奖励设计的稳健性，但也揭示了 RL 无法克服预训练知识不足的局限性。

## Further Thoughts

论文揭示了强化学习在 LLM 中的作用更多是放大预训练中已有的能力，而非从零学习新能力，这提示我们在设计训练策略时，预训练数据的领域覆盖至关重要；密集奖励通过知识蒸馏提供细粒度反馈的思路，是否可以推广到其他战略推理领域（如多智能体博弈或商业决策）？此外，是否可以通过混合方法（结合传统象棋引擎和 LLM）或多阶段训练（先强化规则理解，再进行战略推理）来弥补 LLM 在领域知识上的不足？