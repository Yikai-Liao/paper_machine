---
title: "Improved Algorithms for Differentially Private Language Model Alignment"
pubDatetime: 2025-05-13T16:18:59+00:00
slug: "2025-05-dp-alignment-optimization"
type: "arxiv"
id: "2505.08849"
score: 0.7052659604412173
author: "grok-3-latest"
authors: ["Keyu Chen", "Hao Tang", "Qinglin Liu", "Yizhao Xu"]
tags: ["LLM", "Differential Privacy", "Alignment", "Optimization", "Privacy Budget"]
institution: ["Peking University"]
description: "本文提出了一种统一的隐私保护对齐框架和新型优化器 DP-ADAMW，显著提升了大型语言模型在差分隐私约束下的对齐性能，尤其在 DPO 方法上效果最佳。"
---

> **Summary:** 本文提出了一种统一的隐私保护对齐框架和新型优化器 DP-ADAMW，显著提升了大型语言模型在差分隐私约束下的对齐性能，尤其在 DPO 方法上效果最佳。 

> **Keywords:** LLM, Differential Privacy, Alignment, Optimization, Privacy Budget

**Authors:** Keyu Chen, Hao Tang, Qinglin Liu, Yizhao Xu

**Institution(s):** Peking University


## Problem Background

大型语言模型（LLMs）在对齐（alignment）过程中需要使用用户反馈数据，这些数据往往包含敏感信息，存在隐私泄露风险。
已有研究表明，模型可能记住并重现训练数据中的个人信息，而现有差分隐私（Differential Privacy, DP）与对齐技术结合的方法性能受限，尤其在隐私预算较小时效果不佳。
论文试图解决如何在提供严格隐私保证的同时，实现高质量的语言模型对齐这一关键问题。

## Method

*   **统一框架:** 提出一个统一的隐私保护对齐框架，将现有的对齐技术（如 Direct Preference Optimization, DPO 和 Reinforcement Learning from Human Feedback, RLHF）纳入其中，通过多阶段损失函数最小化实现对齐，同时在每个阶段应用差分隐私机制。
*   **新型优化器 DP-ADAMW:** 开发了一种新的差分隐私优化器 DP-ADAMW，结合 DP-ADAM 的自适应学习率和 ADAMW 的解耦权重衰减机制。具体步骤包括：
    *   在每次迭代中，采样小批量数据计算损失，并对梯度进行裁剪（clipping）以限制单个样本的影响。
    *   添加高斯噪声到裁剪后的梯度，确保满足差分隐私条件。
    *   使用噪声梯度更新一阶动量（momentum）和二阶动量（second moment），并通过偏置校正和权重衰减调整参数更新方向。
*   **应用于对齐技术:** 将 DP-ADAMW 应用于 DPO 和 RLHF 的多阶段训练过程（如监督微调、奖励模型训练、策略优化），在每个阶段保护数据隐私，同时尽量维持模型性能。
*   **隐私分析:** 通过保守的隐私预算累积分析，确保整个训练过程满足差分隐私保证，同时通过数据集分阶段划分减少隐私成本。

## Experiment

*   **有效性:** 实验结果表明，DP-ADAMW 结合 DPO 在中度隐私预算（ε=2-5）下表现最佳，例如在 LLAMA-8B 上，ε=3 时奖励分数达到 1.8814，相比 DP-SGD 的 1.6861 提升约 11.6%，相比非隐私设置（ε=∞）仅略有下降。
*   **全面性:** 实验设置覆盖多个模型（LLAMA-8B, GPT-2, DeepSeek-LLM-7B-Chat）、不同优化器（DP-ADAMW, DP-ADAM, DP-SGD）和对齐方法（DPO, PPO），并在隐私预算从 0 到 ∞ 的范围内测试，全面评估了隐私与性能的权衡。
*   **模型规模影响:** 较大模型（如 LLAMA-8B）对隐私噪声的鲁棒性更强，性能优于小型模型（如 GPT-2），表明模型容量对隐私保护对齐至关重要。
*   **不足:** 在严格隐私预算（ε<2）下，性能仍明显低于非隐私设置，说明在极端隐私约束下仍有改进空间。
*   **计算开销:** 差分隐私机制引入了额外的计算负担，尤其在大型模型和严格隐私预算下，训练时间和资源需求增加。

## Further Thoughts

论文中识别隐私预算临界点（ε0）的分析方法非常有启发性，提示是否可以在训练过程中动态调整隐私预算，例如初期使用较宽松预算加速收敛，后期收紧以增强保护。
此外，大型模型对隐私噪声的鲁棒性启发了我思考是否可以通过模型压缩或混合架构设计，在资源受限场景下模拟大型模型的鲁棒性，从而降低计算成本并推广隐私保护对齐技术。