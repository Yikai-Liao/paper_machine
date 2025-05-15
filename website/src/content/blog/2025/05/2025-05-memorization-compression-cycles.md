---
title: "Memorization-Compression Cycles Improve Generalization"
pubDatetime: 2025-05-13T16:37:54+00:00
slug: "2025-05-memorization-compression-cycles"
type: "arxiv"
id: "2505.08727"
score: 0.7822804953750957
author: "grok-3-latest"
authors: ["Fangyuan Yu"]
tags: ["LLM", "Representation Compression", "Pre-Training", "Generalization", "Information Bottleneck"]
institution: ["Temus"]
description: "本文通过理论和实验证明压缩内部表示可显著提升大型语言模型的泛化能力，并提出 Gated Phase Transition (GAPT) 算法，通过动态切换记忆与压缩阶段实现这一目标。"
---

> **Summary:** 本文通过理论和实验证明压缩内部表示可显著提升大型语言模型的泛化能力，并提出 Gated Phase Transition (GAPT) 算法，通过动态切换记忆与压缩阶段实现这一目标。 

> **Keywords:** LLM, Representation Compression, Pre-Training, Generalization, Information Bottleneck

**Authors:** Fangyuan Yu

**Institution(s):** Temus


## Problem Background

大型语言模型（LLM）的泛化能力传统上依赖于数据和参数规模的扩展，但高质量数据资源已接近枯竭，且后训练方法如强化学习（RLVR）未能有效激励新的推理模式。
论文提出，内部表示的熵（representation entropy）是影响泛化能力的另一关键维度，旨在探索如何通过压缩内部表示，在不依赖大规模数据的情况下提升模型的泛化能力。

## Method

*   **理论框架:** 基于信息瓶颈（Information Bottleneck, IB）理论，提出信息瓶颈语言建模（IBLM）目标，将语言建模重构为一个约束优化问题：即在保持最佳预测性能（最小化交叉熵损失）的前提下，最小化内部表示的熵（representation entropy）。通过定理证明了 IBLM 与经典 IB 框架在语言建模场景下的等价性。
*   **具体算法:** 提出 Gated Phase Transition (GAPT) 训练算法，通过动态切换‘记忆阶段’（仅优化交叉熵损失以捕获数据）和‘压缩阶段’（同时优化交叉熵和矩阵基熵 MBE 以压缩表示）来模拟生物学习中的‘觉醒学习-睡眠巩固’循环。
*   **实现细节:** GAPT 使用基于耐心（patience-based）的门控机制，根据训练信号（如交叉熵和 MBE 的变化）决定阶段切换，避免表示崩塌（representation collapse）。在压缩阶段，MBE 作为表示熵的度量，通过对模型中间层的表示矩阵计算 Gram 矩阵的奇异值分布熵来实现量化。
*   **创新点:** GAPT 不依赖固定的 Lagrangian 优化（如 CE + λ*MBE），而是动态调整训练目标，确保压缩不会损害预测性能，同时借鉴生物学习循环的机制设计训练过程。

## Experiment

*   **预训练效果:** 在 FineWeb 数据集上，GAPT 相比基线（仅优化交叉熵）降低了 4.8% 的交叉熵损失，同时在目标层上平均降低了 70.5% 的 MBE，表明其在保持预测性能的同时显著压缩了内部表示。
*   **泛化能力:** 在算术任务中，GAPT 降低了 35% 的域外（OOD）熵和 47% 的平均 MBE，显示出更强的泛化能力，尤其是在从 1-3 位数乘法训练到 4-6 位数测试的场景中。
*   **冲突记忆分辨:** 在模拟灾难性遗忘的合成任务中，GAPT 提升了 97% 的表示分离度，降低了 91% 的 MBE，表明其在缓解表示干扰方面有显著效果。
*   **实验合理性与局限:** 实验覆盖了预训练、泛化和冲突记忆三个场景，设置较为全面，数据量和模型规模符合研究标准；但算术任务的 OOD 设置中存在不稳定性（需早停策略），可能限制方法的普适性。

## Further Thoughts

论文将记忆-压缩循环与生物学习中的觉醒-睡眠循环类比，这一视角启发我们是否可以进一步探索其他生物机制（如神经元竞争或遗忘机制）来设计更高效的训练算法；此外，MBE 作为表示熵的度量存在计算复杂性和局限性，未来可以尝试基于互信息的熵估计方法来更精确地捕捉表示压缩；最后，压缩表示带来的泛化能力可能增加模型不可解释性和安全风险，提示我们在泛化能力和模型可控性之间寻找平衡。