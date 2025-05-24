---
title: "DeFTX: Denoised Sparse Fine-Tuning for Zero-Shot Cross-Lingual Transfer"
pubDatetime: 2025-05-21T04:20:30+00:00
slug: "2025-05-denoised-sparse-tuning"
type: "arxiv"
id: "2505.15090"
score: 0.6794334553542066
author: "grok-3-latest"
authors: ["Sona Elza Simon", "Preethi Jyothi"]
tags: ["LLM", "Sparse Fine-Tuning", "Cross-Lingual Transfer", "Denoising", "Low-Resource Languages"]
institution: ["Indian Institute of Technology Bombay"]
description: "D E FT-X 通过在稀疏微调前引入 SVD 去噪步骤，提升了零样本跨语言迁移中稀疏向量的质量，从而在低资源语言任务上取得了优于现有方法的性能。"
---

> **Summary:** D E FT-X 通过在稀疏微调前引入 SVD 去噪步骤，提升了零样本跨语言迁移中稀疏向量的质量，从而在低资源语言任务上取得了优于现有方法的性能。 

> **Keywords:** LLM, Sparse Fine-Tuning, Cross-Lingual Transfer, Denoising, Low-Resource Languages

**Authors:** Sona Elza Simon, Preethi Jyothi

**Institution(s):** Indian Institute of Technology Bombay


## Problem Background

大型预训练语言模型在高资源语言上表现优异，但在低资源语言上的零样本跨语言迁移（Zero-Shot Cross-Lingual Transfer）效果有限，尤其是在缺乏目标语言标注数据的情况下。
现有方法如 LT-SFT 通过稀疏微调生成任务和语言特定的稀疏向量并组合实现迁移，但这些向量可能包含噪声或无关信息，影响迁移性能。
论文旨在通过改进稀疏微调向量的质量，提升低资源语言上的任务表现。

## Method

*   **核心思想:** 在稀疏微调前对模型权重进行去噪，减少噪声干扰，从而提高稀疏子网络的质量，增强零样本跨语言迁移效果。
*   **具体步骤:**
    *   **权重差异计算:** 首先对预训练模型进行全微调，计算预训练权重与全微调权重之间的差异（ΔW），作为后续处理的起点。
    *   **去噪处理:** 对 ΔW 中的每个权重矩阵应用奇异值分解（SVD），将其分解为低阶（高奇异值，保留主要信号）和高阶（低奇异值，视为噪声）成分；对高阶成分进行幅度剪枝，仅保留少量有用信息后与低阶成分重组，形成去噪后的权重矩阵。
    *   **稀疏子网络选择:** 在去噪后的 ΔW 上应用幅度剪枝（Magnitude Pruning），选择 Top-k 参数形成稀疏子网络，并重置其他参数为预训练值。
    *   **稀疏微调:** 仅对选定的稀疏子网络参数进行微调，生成语言特定和任务特定的稀疏微调向量（SFTs）。
    *   **向量组合:** 通过简单加法将语言特定和任务特定向量与预训练模型组合，形成目标语言任务模型，用于零样本推理。
*   **创新点:** 相比 LT-SFT 直接基于幅度剪枝选择子网络，D E FT-X 引入 SVD 去噪步骤，减少了权重中的噪声干扰，提升了稀疏向量的有效性和鲁棒性。
*   **优势:** 方法保持了参数高效性，仅微调少量参数，同时通过去噪减少了负干扰，尤其适用于极低资源语言场景。

## Experiment

*   **有效性:** 在 XLM-R BASE 模型上，D E FT-X 在 NusaX（情感分析）数据集的平均 F1 分数为 81.4，优于 LT-SFT（80.2）和 MAD-X（75.8）；在 AmericasNLI（自然语言推理）数据集的平均准确率为 51.3，略优于 LT-SFT（51.0）和 MAD-X（49.5）。在 XLM-R LARGE 模型上，D E FT-X 也在多个配置下超越 LT-SFT。
*   **显著性分析:** 虽然在某些任务（如 AmericasNLI）上的提升幅度较小（约 0.3%），但在极低资源语言场景下，这种微小提升仍具意义，尤其在 NusaX 部分语言（如 Minangkabau）上 F1 分数提升超过 2%。
*   **实验设置合理性:** 实验覆盖了 NusaX（5 种语言）和 AmericasNLI（10 种语言）两个低资源数据集，语言均为预训练未见，符合零样本设定；测试了不同秩选择策略（90% 方差和统一秩）及两种模型规模（XLM-R BASE 和 LARGE），设置全面；消融实验验证了去噪、稀疏性和微调各步骤的必要性。
*   **局限性:** XLM-R LARGE 在 AmericasNLI 上的表现不如 BASE 模型，可能由于大型模型对高资源语言的偏见更强，提示方法对模型规模和预训练数据分布的敏感性。

## Further Thoughts

D E FT-X 的去噪思想可以通过 SVD 之外的其他低秩近似方法（如主成分分析）进一步探索，是否能更高效地提取主要信号？此外，去噪和稀疏微调是否可以应用于多模态模型（如图像-文本任务），以减少模态间噪声干扰？秩选择目前依赖手动调参，是否可以通过自适应算法或元学习动态优化？最后，组合稀疏向量时，是否可以引入加权机制或冲突解决策略，进一步减少语言和任务向量间的负干扰？