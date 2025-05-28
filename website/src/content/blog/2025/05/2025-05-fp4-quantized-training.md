---
title: "FP4 All the Way: Fully Quantized Training of LLMs"
pubDatetime: 2025-05-25T12:14:25+00:00
slug: "2025-05-fp4-quantized-training"
type: "arxiv"
id: "2505.19115"
score: 0.7664754899113291
author: "grok-3-latest"
authors: ["Brian Chmiel", "Maxim Fishman", "Ron Banner", "Daniel Soudry"]
tags: ["LLM", "Quantization", "Low Precision", "Training Efficiency"]
institution: ["Intel, Israel", "Department of Electrical and Computer Engineering, Technion, Haifa, Israel"]
description: "本文首次实现了大型语言模型在 FP4 精度下的完全量化训练，通过优化格式设计、分阶段舍入策略和理论阈值指导，在保持与 BF16 基线相当性能的同时显著提升了训练效率潜力。"
---

> **Summary:** 本文首次实现了大型语言模型在 FP4 精度下的完全量化训练，通过优化格式设计、分阶段舍入策略和理论阈值指导，在保持与 BF16 基线相当性能的同时显著提升了训练效率潜力。 

> **Keywords:** LLM, Quantization, Low Precision, Training Efficiency

**Authors:** Brian Chmiel, Maxim Fishman, Ron Banner, Daniel Soudry

**Institution(s):** Intel, Israel, Department of Electrical and Computer Engineering, Technion, Haifa, Israel


## Problem Background

随着大型语言模型（LLMs）规模的快速增长，训练和推理所需的计算资源与内存带宽成为主要瓶颈。
传统的 BF16 格式在精度与效率间取得平衡，但仍无法满足超大规模模型的需求，因此研究转向更低精度的 FP4 格式以进一步提升效率。
然而，FP4 的动态范围有限，容易导致量化误差，影响训练稳定性和模型精度，亟需探索在 FP4 精度下实现完全量化训练（Fully Quantized Training, FQT）的可行性。

## Method

*   **核心目标:** 实现大型语言模型在 FP4 精度下的完全量化训练，覆盖权重、激活值和梯度，同时保持与 BF16 基线相当的性能。
*   **FP4 格式设计:** 系统对比了 MXFP4 和 NVFP4 两种格式，发现 NVFP4（块大小为 16，缩放格式为 E4M3）在动态范围和精度间取得最佳平衡，验证了 NVIDIA Blackwell 硬件设计的合理性。
*   **分阶段舍入策略:** 在前向传播中采用‘最近舍入’（Round-to-Nearest, RtN）以确保计算确定性，在反向传播和更新阶段采用‘随机舍入’（Stochastic Rounding, SR）以减少量化偏差，提升训练稳定性。
*   **理论阈值指导:** 提出当全精度梯度标准差低于量化噪声标准差的 √3 倍时，训练效果显著下降，基于此建议在训练后期通过量化感知微调（Quantization Aware Finetuning, QAF）切换到更高精度（如 BF16）以提升信号噪声比。
*   **端到端实现:** 在 256 个 Intel Gaudi2 加速器上，使用 NVFP4 格式和上述策略，成功训练 7B 参数的 Llama2 模型，并在 QAF 阶段消除与 BF16 的损失差距。

## Experiment

*   **训练效果:** 在 Red Pajama 数据集（约 2000 亿 token）上训练 Llama2 7B 模型，FP4 训练的损失与 BF16 基线有轻微差距，但通过短暂的 QAF 阶段（前向仍为 FP4，反向和更新为 BF16）完全消除差距。
*   **下游任务性能:** 在多个零样本任务（如 HellaSwag, Winogrande, BoolQ 等）上，FP4 模型在 QAF 后与 BF16 基线性能相当，表明效率提升未以质量为代价。
*   **实验全面性:** 系统测试了不同块大小（8 到 128）和缩放格式（E1M6 到 E8M0），验证了 NVFP4 的优越性；对比了不同舍入模式组合，确认分阶段舍入策略有效。
*   **局限性:** 由于 Intel Gaudi2 硬件不支持原生 FP4，实验通过模拟实现，无法直接测量加速和能效收益，仅基于 FP8 研究估算约 35-40% 的训练时间加速（相比 FP8），存在不确定性。

## Further Thoughts

论文提出的梯度信号与量化噪声比例阈值（√3 倍）为低精度训练提供了一个量化切换点，未来可探索动态精度调整策略，根据训练阶段实时监控信号噪声比优化资源分配；此外，分阶段舍入策略提示不同计算阶段对量化误差的敏感性不同，可在其他低精度场景中定制量化策略；最后，NVFP4 格式与硬件支持的契合表明算法与硬件协同设计的重要性，未来研究可更深度参与硬件架构优化。