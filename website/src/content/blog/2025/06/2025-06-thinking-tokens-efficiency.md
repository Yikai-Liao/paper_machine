---
title: "Do Thinking Tokens Help or Trap? Towards More Efficient Large Reasoning Model"
pubDatetime: 2025-06-30T13:30:33+00:00
slug: "2025-06-thinking-tokens-efficiency"
type: "arxiv"
id: "2506.23840"
score: 0.6727389933121061
author: "grok-3-latest"
authors: ["Bowen Ding", "Yuhan Chen", "Futing Wang", "Lingfeng Ming", "Tao Lin"]
tags: ["LLM", "Reasoning", "Sampling", "RLHF", "Post-Training"]
institution: ["Zhejiang University", "Boston University", "ByteDance", "School of Engineering, Westlake University", "Research Center for Industries of the Future, Westlake University"]
description: "本文提出双策略偏好优化（DuP-PO）算法，通过强化学习动态控制思考标记生成，显著提升大型推理模型的性能和 token 效率。"
---

> **Summary:** 本文提出双策略偏好优化（DuP-PO）算法，通过强化学习动态控制思考标记生成，显著提升大型推理模型的性能和 token 效率。 

> **Keywords:** LLM, Reasoning, Sampling, RLHF, Post-Training

**Authors:** Bowen Ding, Yuhan Chen, Futing Wang, Lingfeng Ming, Tao Lin

**Institution(s):** Zhejiang University, Boston University, ByteDance, School of Engineering, Westlake University, Research Center for Industries of the Future, Westlake University


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）在复杂任务中表现出色，但常因‘过度思考’（overthinking）问题生成冗长响应，尤其是在简单任务上，包含大量‘思考标记’（thinking tokens，如‘wait’‘however’），这些标记触发不必要的高级推理行为（如反思、回溯），导致计算资源浪费和效率低下。
作者通过初步研究发现，这些思考标记可能并非解决问题所必需，甚至在有限 token 预算内阻碍正确推理，称之为‘思考陷阱’（thinking trap），因此提出研究思考标记的作用并优化模型效率。

## Method

*   **核心思想:** 通过强化学习动态调节思考标记的生成，避免‘思考陷阱’，在保持推理质量的同时提升 token 效率。
*   **具体实现:** 提出‘双策略偏好优化’（Dual Policy Preference Optimization, DuP-PO）算法，基于 GRPO（Group-based Reward Policy Optimization）改进，包含以下关键组件：
    *   **双策略采样（Dual-Policy Sampling）:** 使用两种策略生成推理轨迹：正常策略（normal policy）保留思考标记，修正策略（rectified policy）通过将思考标记的 logit 值设为负无穷来阻止其生成。训练时从两种策略各采样一部分轨迹，确保模型接触到包含和不包含思考标记的响应，学习区分有益与有害的思考行为。
    *   **token 级优势缩放（Token-Level Advantage Scaling）:** 针对不同 token 应用不同的优势缩放因子，对于来自修正策略的优选轨迹（简洁推理）增强奖励，对于正常策略中非优选轨迹的思考标记加强抑制，同时对其他 token 保持标准处理，实现细粒度的概率调整。
    *   **策略塑造（Policy Shaping）:** 针对思考标记的高预测概率问题，校准旧策略概率，确保抑制思考标记的梯度不会因剪切限制而丢失，提供稳定的学习信号。
*   **关键特点:** 不完全消除思考标记，而是根据任务需求动态控制其使用；方法轻量，仅需少量训练步骤即可显著提升效率。

## Experiment

*   **有效性:** 在 DeepSeek-R1-Distill-Qwen-1.5B 模型上，DuP-PO 平均性能提升 4.0 个百分点，同时 token 使用量减少 15.4%。在简单任务（如 MATH500）上提升最显著（3.5 百分点，token 减少 24.7%），在较难任务（如 AIME24, AIME25）上性能提升稍小（1.3-3.2 百分点，token 减少 8.3%-10.7%），显示出对任务复杂度的适应性。
*   **与基线对比:** 相比无训练方法 ThinkTokenPenalty（token 减少 30.6%，准确率不变），DuP-PO 提升 3.9 个百分点且保持显著效率；相比 GRPO，DuP-PO 在更少训练步数（80 vs 90）下提升 1.3 个百分点，token 消耗更低（5162 vs 5724）。
*   **实验设置合理性:** 实验覆盖六个数学推理基准数据集（AIME24, AIME25, AMC, Minerva, OlympiadBench, MATH500），数据选择聚焦中等难度且易陷入思考陷阱的问题，多次推理运行减少随机性，参数设置细致，整体设计全面合理。
*   **局限性:** 实验主要基于 1.5B 参数模型，未测试更大规模模型，普适性有待验证。

## Further Thoughts

论文揭示了思考标记作用的上下文依赖性，启发我们设计‘自适应思考深度’机制，根据任务难度动态调整推理行为；双策略采样的对比学习思想可扩展至多模态推理或对话系统，通过对比不同行为模式优化决策；token 级细粒度控制展示了强化学习在行为微调上的潜力，未来可探索调控其他 token 属性（如情感、逻辑性）以适配特定场景。