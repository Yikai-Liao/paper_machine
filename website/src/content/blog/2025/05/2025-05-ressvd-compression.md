---
title: "ResSVD: Residual Compensated SVD for Large Language Model Compression"
pubDatetime: 2025-05-26T15:14:54+00:00
slug: "2025-05-ressvd-compression"
type: "arxiv"
id: "2505.20112"
score: 0.6744707622359332
author: "grok-3-latest"
authors: ["Haolei Bai", "Siyong Jian", "Tuo Liang", "Yu Yin", "Huan Wang"]
tags: ["LLM", "Model Compression", "Low-Rank Approximation", "Residual Compensation", "Error Propagation"]
institution: ["Westlake University", "Nanyang Technological University", "Nanjing University", "Case Western Reserve University"]
description: "ResSVD 提出了一种后训练 SVD 压缩方法，通过残差补偿和部分层压缩显著减少大型语言模型的截断损失和误差传播，全面优于现有基线方法。"
---

> **Summary:** ResSVD 提出了一种后训练 SVD 压缩方法，通过残差补偿和部分层压缩显著减少大型语言模型的截断损失和误差传播，全面优于现有基线方法。 

> **Keywords:** LLM, Model Compression, Low-Rank Approximation, Residual Compensation, Error Propagation

**Authors:** Haolei Bai, Siyong Jian, Tuo Liang, Yu Yin, Huan Wang

**Institution(s):** Westlake University, Nanyang Technological University, Nanjing University, Case Western Reserve University


## Problem Background

大型语言模型（LLMs）因其巨大的参数量和内存需求，难以在资源受限的环境中部署，亟需有效的压缩策略。
现有的奇异值分解（SVD）方法在压缩模型权重矩阵时，忽略了截断过程中产生的残差矩阵，导致较大的截断损失；同时，对所有层进行压缩会引发误差传播，严重影响性能。
ResSVD 旨在通过改进 SVD 压缩方法，减少截断损失并缓解误差传播问题，从而实现更高效的模型压缩。

## Method

*   **核心思想:** ResSVD 提出了一种后训练的 SVD 压缩方法，通过利用截断过程中的残差矩阵减少损失，并选择性地压缩模型的最后几层以缓解误差传播。
*   **残差补偿（Residual Compensation）:** 
    *   传统 SVD 截断直接丢弃小奇异值对应的部分，导致较大的截断损失。ResSVD 采用两阶段截断策略：
        1. 首先对原始权重矩阵 W 进行 SVD 截断，得到低秩近似 W_r1（秩为 r1）。
        2. 计算残差矩阵 R = W - W_r1，再对 R 进行第二次 SVD 截断，得到低秩近似 R_r2（秩为 r2，满足 r1 + r2 = r）。
        3. 最终压缩矩阵为 W_hat_r = W_r1 + R_r2。
    *   理论分析（基于 Eckart-Young-Mirsky 定理）证明，这种方法得到的压缩矩阵比直接截断更接近原始矩阵，截断损失更小。
*   **部分层压缩（Partial-Layer Compression）:** 
    *   观察到模型早期层的压缩会导致误差在后续层中累积传播，ResSVD 提出在固定整体压缩比下，仅压缩模型的最后几层，保持早期层不变。
    *   通过层级误差分析，选择最后 k 层进行压缩（k 根据整体压缩比和层数计算），从而显著降低整体误差。
*   **实现细节:** 使用校准数据（如 WikiText-2）进行数据白化（data whitening）以优化截断过程，所有操作均为后训练，无需重新训练模型。

## Experiment

*   **有效性:** ResSVD 在 20% 到 60% 的压缩比下，显著优于基线方法（如 SVD、ASVD、SVD-LLM、AdaSVD）。在语言建模任务（WikiText-2、PTB、C4）中，困惑度（perplexity）降低幅度最高达 41%（C4，50% 压缩比）；在零样本推理任务中，平均准确率提升最高达 9%。
*   **可扩展性与泛化性:** 在多个模型家族（LLaMA、OPT、Mistral、Vicuna）和更大规模模型（LLaMA-30B、OPT-30B）上，ResSVD 一致表现出色，例如在 Mistral-7B 上困惑度降低高达 71%（WikiText-2，30% 压缩比）。
*   **实验设置合理性:** 实验覆盖多种压缩比、模型规模和任务类型（语言建模和推理），使用统一的校准数据和实现框架（PyTorch、Transformers），对比了多个基线方法。消融研究验证了残差补偿和部分层压缩的独立贡献。
*   **效率提升:** 在 NVIDIA A100 GPU 上，ResSVD 压缩模型的推理吞吐量高于基线，尤其在大批量处理时速度提升更显著，表明其在实际部署中的潜力。

## Further Thoughts

残差补偿的概念可以推广到其他压缩技术（如量化或剪枝），通过迭代补偿损失进一步提升效果；部分层压缩策略启发我们根据层级误差敏感性动态选择压缩层，而非固定最后几层；校准数据对性能的影响提示可以设计自适应数据选择机制，根据目标任务优化压缩效果。