---
title: "Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks?"
pubDatetime: 2025-05-19T08:57:08+00:00
slug: "2025-05-lora-robustness-attacks"
type: "arxiv"
id: "2505.12871"
score: 0.5405647487414988
author: "grok-3-latest"
authors: ["Zi Liang", "Haibo Hu", "Qingqing Ye", "Yaxin Xiao", "Ronghua Li"]
tags: ["LLM", "Low Rank Adaptation", "Fine-Tuning", "Robustness", "Training-Time Attacks"]
institution: ["The Hong Kong Polytechnic University"]
description: "本文通过理论框架和实验分析揭示了低秩适应（LoRA）在训练时攻击下的鲁棒性特性，发现其对后门攻击更鲁棒但对无目标数据投毒更脆弱，并提出秩和初始化方差是影响安全性的关键因素。"
---

> **Summary:** 本文通过理论框架和实验分析揭示了低秩适应（LoRA）在训练时攻击下的鲁棒性特性，发现其对后门攻击更鲁棒但对无目标数据投毒更脆弱，并提出秩和初始化方差是影响安全性的关键因素。 

> **Keywords:** LLM, Low Rank Adaptation, Fine-Tuning, Robustness, Training-Time Attacks

**Authors:** Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Ronghua Li

**Institution(s):** The Hong Kong Polytechnic University


## Problem Background

大型语言模型（LLMs）在参数高效微调（PEFT）中广泛采用低秩适应（LoRA）方法以降低计算成本，但其在训练时攻击（如数据投毒和后门攻击）下的安全风险尚未被充分研究。
论文旨在探究 LoRA 微调是否比全参数微调（Full Fine-Tuning, FF）在训练时攻击下更脆弱，揭示其低秩结构对训练时鲁棒性（Training-Time Robustness, TTR）的影响，以填补现有研究在 LoRA 内在安全漏洞上的空白。

## Method

*   **核心思想:** 提出一个理论分析框架，通过神经切线核（Neural Tangent Kernel, NTK）和信息几何（Information Geometry, IG）研究 LoRA 的低秩结构对训练时鲁棒性（TTR）的影响。
*   **具体步骤:**
    *   **定义 TTR:** 通过比较参数更新在干净数据集和受扰数据集上的差异，量化模型对训练时攻击的敏感性，作为鲁棒性指标。
    *   **NTK 简化训练动态:** 利用 NTK 理论将复杂的训练过程简化为梯度相似性度量，假设网络宽度趋于无穷大时训练动态可用确定的核函数表示，从而避免直接分析参数更新的动态复杂性。
    *   **信息几何分析结构特性:** 引入 Fisher 信息矩阵和 Rényi 熵，分析 LoRA 低秩结构导致的信息几何特性（如信息位和曲率），并将其与 TTR 关联，揭示结构对鲁棒性的影响。
    *   **理论推导与比较:** 推导 LoRA 和全参数微调的 NTK 表达式，比较两者的信息位（Information Bits, IB）和信息几何曲率，得出 LoRA 由于低秩结构导致信息几何更平滑，对后门攻击更鲁棒，但对无目标数据投毒更脆弱。
*   **关键创新:** 该方法从理论上揭示了 LoRA 的低秩结构如何影响其训练时行为，而不仅仅依赖实验观察，为理解高效微调的安全性提供了新视角。

## Experiment

*   **有效性:** 实验结果支持理论分析，LoRA 在无目标数据投毒攻击（UPA）下比全参数微调（FF）表现出更明显的性能下降（例如在 QNLI 和 QQP 数据集上，投毒率 0.3 时准确率差距显著）；而在后门攻击（BPA）下，LoRA 表现出更强鲁棒性（例如在 SST-2 和 QQP 数据集上准确率提升高达 30%）。
*   **合理性:** 实验在 GLUE 基准数据集上使用 BERT-large 模型，覆盖多种攻击类型和投毒率（0.05 到 0.35），重复五次以报告均值和标准差，确保结果可靠性；此外，测试了秩（rank）和初始化方差的影响，验证了理论推导的关键参数作用。
*   **局限性:** 实验主要集中于分类任务和 BERT 模型，尽管附录中扩展到生成模型，但覆盖范围有限，可能无法完全代表所有 LLM 场景。

## Further Thoughts

论文通过信息几何量化模型结构对训练时攻击鲁棒性的影响，这一方法启发我们探索其他高效微调技术（如 Adapter 或 Prefix Tuning）的安全性；此外，LoRA 的秩和初始化方差在不同攻击类型下的权衡效应，提示可以设计自适应超参数调整策略或自动化优化工具，根据任务和安全需求动态平衡性能与鲁棒性。