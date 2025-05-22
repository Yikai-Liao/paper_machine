---
title: "Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training"
pubDatetime: 2025-05-19T21:27:33+00:00
slug: "2025-05-scaling-laws-llm-training"
type: "arxiv"
id: "2505.13738"
score: 0.8017796293063354
author: "grok-3-latest"
authors: ["Shane Bergsma", "Nolan Dey", "Gurpreet Gosal", "Gavia Gray", "Daria Soboleva", "Joel Hestness"]
tags: ["LLM", "Scaling Laws", "Hyperparameter Tuning", "Batch Size", "Weight Decay", "Pre-Training", "Compute Efficiency", "Data Parallelism", "Over-Training"]
institution: ["Cerebras Systems"]
description: "本文通过大规模实验建立了 LLM 预训练中权重衰减和批大小的缩放法则，揭示了 AdamW 时间尺度随 TPP 幂律下降及批大小随数据集规模幂律增长的规律，为高效训练提供了预测性指导。"
---

> **Summary:** 本文通过大规模实验建立了 LLM 预训练中权重衰减和批大小的缩放法则，揭示了 AdamW 时间尺度随 TPP 幂律下降及批大小随数据集规模幂律增长的规律，为高效训练提供了预测性指导。 

> **Keywords:** LLM, Scaling Laws, Hyperparameter Tuning, Batch Size, Weight Decay, Pre-Training, Compute Efficiency, Data Parallelism, Over-Training

**Authors:** Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, Joel Hestness

**Institution(s):** Cerebras Systems


## Problem Background

在大型语言模型（LLM）预训练中，超参数如学习率（η）、权重衰减（λ）和批大小（B）的调优对性能至关重要，但大规模训练的高计算成本使得传统试错方法不可行。
本文旨在解决如何在模型规模（N）、数据集规模（D）和批大小（B）变化时，系统性地预测最优超参数设置，以优化计算效率和训练时间的平衡，减少大规模训练中的调参负担。

## Method

*   **AdamW 时间尺度（τ_EMA）缩放法则**：定义 τ_EMA = B/(ηλD) 作为 AdamW 优化器权重更新的时间尺度，发现其最优值随每参数令牌数（TPP = D/N）呈幂律下降，通过这一规律预测不同规模下的最优权重衰减 λ。
*   **批大小缩放法则**：研究最优批大小（B_opt，达到最低损失的批大小）和临界批大小（B_crit，数据并行效率下降的阈值），提出两者随数据集规模 D 呈幂律增长，而非依赖于计算量（FLOPs）或损失（Loss）。
*   **最大更新参数化（µP）框架**：利用 µP 通过小规模代理模型调优学习率 η，并将其转移到大规模模型，确保学习率随模型宽度变化的稳定性，同时优先调整 λ 而非 η 以维持 τ_EMA 的最优值。
*   **Pareto 最优配置分析**：基于缩放法则，探索在计算成本和训练时间双重目标下的最优模型规模（N）和数据规模（D）配置，分析过度训练（Over-Training）与计算最优（Compute-Optimal）模型的权衡。

## Experiment

*   **有效性**：τ_EMA 的幂律缩放关系（R²=0.975）在不同规模模型上表现出色，即使在计算规模相差三个数量级的测试中也能准确预测最优权重衰减 λ；B_opt 和 B_crit 随 D 的幂律关系（R² 分别为 0.984 和 0.940）优于先前假设，且与近期研究结果一致。
*   **实验设置**：实验覆盖模型规模从 111M 到 3.3B 参数，TPP 从 20 到 1280，批大小从 63 到 8064 序列，使用 SlimPajama 数据集，验证集固定为 11 亿令牌，总共训练约 400 个模型，数据点丰富且设置合理。
*   **局限性**：主要基于 AdamW 优化器和单一学习率调度（线性衰减），未充分探索其他优化器或动态批大小策略；小批大小下的损失退化现象未完全解释，可能需要进一步调优其他超参数。

## Further Thoughts

论文揭示了调整权重衰减 λ 比学习率 η 更有效的策略，这可能启发未来优化器设计中对权重衰减作用的重新思考，尤其是在大规模训练中如何通过 λ 平衡训练稳定性和效率；此外，过度训练的小模型在时间和并行性优先场景下的 Pareto 优势，提示在资源受限或快速迭代环境中可优先考虑小模型高 TPP 的训练策略；最后，批大小缩放主要依赖于数据集规模 D 的发现，建议未来训练计划设计时应更关注数据规模而非单纯模型参数量或计算预算。