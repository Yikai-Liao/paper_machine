---
title: "AdaptThink: Reasoning Models Can Learn When to Think"
pubDatetime: 2025-05-19T17:50:52+00:00
slug: "2025-05-adaptthink-reasoning-efficiency"
type: "arxiv"
id: "2505.13417"
score: 0.8689223945654835
author: "grok-3-latest"
authors: ["Jiajie Zhang", "Nianyi Lin", "Lei Hou", "Ling Feng", "Juanzi Li"]
tags: ["LLM", "Reasoning", "Efficiency", "Reinforcement Learning", "Adaptive Strategy"]
institution: ["Tsinghua University"]
description: "本文提出 *AdaptThink*，一种基于强化学习的算法，使推理模型根据问题难度自适应选择思维模式，在显著降低推理成本的同时提升性能。"
---

> **Summary:** 本文提出 *AdaptThink*，一种基于强化学习的算法，使推理模型根据问题难度自适应选择思维模式，在显著降低推理成本的同时提升性能。 

> **Keywords:** LLM, Reasoning, Efficiency, Reinforcement Learning, Adaptive Strategy

**Authors:** Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, Juanzi Li

**Institution(s):** Tsinghua University


## Problem Background

大型推理模型（Large Reasoning Models, LRMs）通过生成详细的思维链（Chain of Thought）在复杂任务上表现出色，但这种长时间的思考过程带来了高昂的推理成本和延迟，尤其在简单任务上显得冗余。
现有方法多集中于缩短响应长度，但未考虑问题难度的差异，导致资源分配不合理。
论文受到‘NoThinking’方法的启发，发现对于简单任务，直接生成最终答案在性能和效率上可能更优，因此提出研究问题：能否让模型根据问题难度自适应选择‘思考’（Thinking）或‘不思考’（NoThinking）模式，以优化效率与性能的权衡？

## Method

*   **核心思想:** 提出 *AdaptThink*，一种基于强化学习（RL）的算法，训练推理模型根据问题难度自适应选择‘Thinking’或‘NoThinking’模式，以在保持甚至提升性能的同时显著降低推理成本。
*   **具体实现:** 方法包含两个关键组件：
    *   **约束优化目标:** 设计一个优化目标，鼓励模型优先选择‘NoThinking’模式以减少计算开销，同时通过惩罚项确保整体准确率不下降。优化目标结合了选择‘NoThinking’的概率和任务奖励（准确率），并使用PPO（Proximal Policy Optimization）风格的损失函数，通过策略梯度方法进行优化。
    *   **重要性采样策略:** 针对初始模型倾向于总是选择‘Thinking’模式导致的冷启动问题，引入重要性采样（Importance Sampling），在训练时平衡‘Thinking’和‘NoThinking’样本的比例，确保模型从一开始就能探索两种模式，并在整个训练过程中维持探索与利用的平衡。
*   **创新点:** 与传统缩短响应长度的方法不同，*AdaptThink* 从根本上让模型学会‘是否需要思考’，实现了更智能的资源分配，且不依赖于特定的模型架构，具有较强的通用性。

## Experiment

*   **有效性:** *AdaptThink* 在减少推理成本方面表现突出，DeepSeek-R1-Distill-Qwen-1.5B 模型平均响应长度减少了 53.0%，7B 模型减少了 40.1%，同时准确率分别提升了 2.4% 和 2.3%，表明方法在效率和性能之间取得了良好平衡。
*   **自适应性:** 实验数据表明，*AdaptThink* 能根据问题难度调整思维模式比例，例如在较简单的 GSM8K 和 MATH500 数据集上更多选择‘NoThinking’模式（比例高达 86.9% 和 76.8%），而在更难的 AIME2024 上更多选择‘Thinking’模式，展现了自适应能力。
*   **对比基线:** 与多种基线方法（如 DPO、OverThink、ModelMerging 等）相比，*AdaptThink* 在准确率和响应长度上均表现最佳，证明了自适应思维模式选择的优越性。
*   **实验设置合理性:** 实验覆盖了不同规模模型（1.5B 和 7B）和不同难度任务（GSM8K、MATH500、AIME2024），评估指标（准确率和响应长度）直接对应研究目标。论文还测试了超参数 *δ* 的影响，展示了方法的鲁棒性。不足之处在于训练数据限于数学领域，但 MMLU 数据集上的测试表明其具有一定的跨领域泛化能力。

## Further Thoughts

自适应资源分配的理念启发了我，是否可以将计算资源动态分配的思想推广到其他 AI 领域，如根据图像复杂度调整模型深度，或在多模态任务中根据输入类型分配注意力？此外，重要性采样解决冷启动问题的策略也具有普适性，可用于其他 RL 场景中初始策略与目标策略差异较大的情况。进一步思考，是否可以设计更细粒度的思维模式（如浅层、中层、深度思考），或在预训练阶段嵌入难度感知模块，而不仅仅依赖后训练调整？