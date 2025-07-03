---
title: "Do Thinking Tokens Help or Trap? Towards More Efficient Large Reasoning Model"
pubDatetime: 2025-06-30T13:30:33+00:00
slug: "2025-06-thinking-tokens-efficiency"
type: "arxiv"
id: "2506.23840"
score: 0.6727389933121061
author: "grok-3-latest"
authors: ["Bowen Ding", "Yuhan Chen", "Futing Wang", "Lingfeng Ming", "Tao Lin"]
tags: ["Large Reasoning Model", "Thinking Tokens", "Token Efficiency", "Reinforcement Learning", "Overthinking"]
institution: ["Zhejiang University", "Boston University", "ByteDance", "Westlake University"]
description: "本文提出双策略偏好优化（DuP-PO）算法，通过精细控制思考标记的使用，显著提升大型推理模型的 token 效率和性能，实现了性能与效率的优越平衡。"
---

> **Summary:** 本文提出双策略偏好优化（DuP-PO）算法，通过精细控制思考标记的使用，显著提升大型推理模型的 token 效率和性能，实现了性能与效率的优越平衡。 

> **Keywords:** Large Reasoning Model, Thinking Tokens, Token Efficiency, Reinforcement Learning, Overthinking

**Authors:** Bowen Ding, Yuhan Chen, Futing Wang, Lingfeng Ming, Tao Lin

**Institution(s):** Zhejiang University, Boston University, ByteDance, Westlake University


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）在复杂问题解决中表现出色，但面临‘过度思考’（overthinking）困境，即在简单任务中生成冗长响应，包含大量‘思考标记’（thinking tokens，如‘wait’‘however’），这些标记触发不必要的高级推理行为（如反思和回溯），导致效率低下，尤其在 token 预算受限时可能损害正确性。
作者通过初步研究发现，这种‘思考陷阱’（thinking trap）并非对问题解决有益，反而浪费计算资源，因此提出核心问题：思考标记是帮助还是阻碍推理过程？

## Method

*   **核心思想:** 提出‘双策略偏好优化’（Dual Policy Preference Optimization, DuP-PO），一种基于强化学习的算法，通过精细控制思考标记的使用，避免‘思考陷阱’，提升 token 效率，同时维持推理性能。
*   **具体实现:** DuP-PO 基于 GRPO（Group-based Reward Policy Optimization）改进，包含以下三个关键创新：
    *   **双策略采样（Dual-Policy Sampling）**：在训练时使用两种策略生成响应轨迹，‘正常策略’（normal policy）允许思考标记生成，‘修正策略’（rectified policy）通过将思考标记的 logits 设置为负无穷来抑制其生成，确保模型接触到含思考标记和无思考标记的两种响应类型，从而学习区分有益与有害的使用场景。
    *   **Token 级优势缩放（Token-Level Advantage Scaling）**：针对不同轨迹来源和 token 类型（是否为思考标记）调整优势值（advantage），例如对无思考标记的优选轨迹增强正向优势，对含思考标记的非优选轨迹放大负向优势，从而精细控制 token 预测概率的更新方向和强度，避免一刀切的处理。
    *   **策略塑造（Policy Shaping）**：通过校准旧策略的概率分布，确保对思考标记的抑制梯度不会因 GRPO 的剪切机制（clipping）被忽略，提供稳定的学习信号，特别是在思考标记预测概率较高时仍能有效抑制其生成。
*   **关键特点:** 不完全消除思考标记，而是让模型学习在何种情况下使用它们是有效的；方法轻量，仅需少量训练步骤即可显著改进模型行为。

## Experiment

*   **有效性:** 在六个数学推理基准数据集（AIME24, AIME25, AMC, Minerva, OlympiadBench, MATH500）上，DuP-PO 基于 DeepSeek-R1-Distill-Qwen-1.5B 模型，仅需 80 个 RL 训练步骤，就实现了平均 4.0 个百分点的性能提升，同时减少 15.4% 的 token 使用，尤其在简单任务（如 MATH500）上效果显著（性能提升 3.5 个百分点，token 减少 24.7%）。
*   **对比性:** 与无训练基线（如 NoThink 和 ThinkTokenPenalty）相比，DuP-PO 性能提升 3.9 个百分点，同时保持显著 token 效率；与 GRPO 相比，DuP-PO 在更少训练步骤（80 vs 90）下性能提升 1.3 个百分点，且推理时 token 消耗更少（5162 vs 5724）。
*   **合理性:** 实验设置全面，涵盖不同难度基准数据集，数据选择（DuPPO-1K）针对中等难度和长响应问题，确保聚焦‘思考陷阱’；通过多次推理运行减少随机性影响，结果显示 DuP-PO 在性能与效率权衡上表现优异。
*   **开销:** 训练开销较小，仅需少量 RL 步骤，推理时主要增加双策略采样的计算成本，但整体仍具高效性。

## Further Thoughts

论文揭示了思考标记的作用并非固定，而是与任务复杂度和上下文密切相关，这一洞察启发我们可以在更广泛场景中探索 token 级别的动态调控策略，例如根据任务难度自适应调整推理深度或 token 使用模式；此外，DuP-PO 的双策略采样机制表明训练数据分布的多样性（同时包含正反例）对模型学习策略性行为至关重要，这一思想可能适用于其他领域的优化问题，如对话系统中的冗余表达抑制或多模态推理中的资源分配优化。