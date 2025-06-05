---
title: "PoLAR: Polar-Decomposed Low-Rank Adapter Representation"
pubDatetime: 2025-06-03T17:58:19+00:00
slug: "2025-06-polar-low-rank-adapter"
type: "arxiv"
id: "2506.03133"
score: 0.7787177401893933
author: "grok-3-latest"
authors: ["Kai Lion", "Liang Zhang", "Bingcong Li", "Niao He"]
tags: ["LLM", "Low-Rank Adaptation", "Parameter Efficiency", "Fine-Tuning", "Optimization"]
institution: ["ETH Zurich"]
description: "本文提出 PoLAR，一种基于极分解的低秩适配参数化方法，通过正交方向矩阵和黎曼优化显著提升大型语言模型微调性能，同时保持参数效率。"
---

> **Summary:** 本文提出 PoLAR，一种基于极分解的低秩适配参数化方法，通过正交方向矩阵和黎曼优化显著提升大型语言模型微调性能，同时保持参数效率。 

> **Keywords:** LLM, Low-Rank Adaptation, Parameter Efficiency, Fine-Tuning, Optimization

**Authors:** Kai Lion, Liang Zhang, Bingcong Li, Niao He

**Institution(s):** ETH Zurich


## Problem Background

大型语言模型（LLM）因参数量巨大，在硬件受限环境下全参数微调不可行，参数高效微调方法如 LoRA 通过低秩更新减少参数量，但其更新矩阵的稳定秩（stable rank）远低于线性代数秩，导致更新方向多样性崩溃（directional diversity collapse），限制了表达能力和微调性能。
本文旨在解决这一问题，通过新的参数化方法充分利用低秩空间，提升模型性能。

## Method

*   **核心思想：** 提出 PoLAR（Polar-Decomposed Low-Rank Adapter Representation），一种基于极分解（polar decomposition）的低秩适配参数化方法，旨在通过正交性约束增加更新方向的多样性，提升稳定秩和表达能力。
*   **参数化方式：** 将低秩更新矩阵分解为两个列正交的方向矩阵（constrained to Stiefel manifolds）和一个无约束的尺度矩阵（scale matrix），这种分解通过正交性避免方向多样性崩溃。
*   **优化策略：** 采用黎曼优化（Riemannian optimization）方法在 Stiefel 流形上优化方向矩阵，同时引入‘着陆算法’（landing algorithm），通过矩阵乘法替代传统回撤操作（retraction），显著提升 GPU 计算效率。
*   **理论优势：** 在矩阵分解的典型问题上，证明 PoLAR 结合黎曼梯度下降（RGD）能实现指数级更快的收敛速度，优于传统 LoRA 的 Burer-Monteiro 参数化。
*   **实现细节：** 初始化方向矩阵为 Stiefel 流形上的随机矩阵，尺度矩阵初始化为零，通过迭代更新动态调整方向和尺度，确保优化过程稳定且高效。

## Experiment

*   **有效性：** 在常识推理（Commonsense Reasoning）、数学问题解决（GSM8K, MATH）和自然语言理解（GLUE）等任务上，PoLAR 显著优于 LoRA 和 DoRA，例如在 Llama-2-7B 上常识推理平均准确率提升约 1-1.5 个百分点，在 Gemma-3-27B 上 MATH 数据集准确率从 41.94% 提升至 42.70%。
*   **稳定秩提升：** PoLAR 显著提高更新矩阵的稳定秩，LoRA 稳定秩通常低于 2，而 PoLAR 随名义秩增加而稳步提升，验证了其更有效利用低秩空间的能力。
*   **计算效率：** 着陆算法在 GPU 上运行时间比传统回撤方法快 3-18 倍，尤其在高秩设置下优势明显。
*   **实验全面性：** 测试覆盖模型规模从 350M 到 27B，包含掩码和自回归模型，任务类型多样，实验设置合理且具有代表性，同时分析了稳定秩动态和层类型差异，数据支持理论假设。

## Further Thoughts

PoLAR 揭示了稳定秩与微调性能的正相关性，启发我们可以在其他参数高效微调方法中探索类似指标，设计新参数化或正则化策略；其正交性约束思想可扩展至计算机视觉或扩散模型微调；此外，算法-硬件协同设计（如着陆算法）提示未来研究应更多关注硬件特性以降低计算成本；最后，研究低秩学习的谱分布动态可能结合随机矩阵理论进一步揭示学习机制。