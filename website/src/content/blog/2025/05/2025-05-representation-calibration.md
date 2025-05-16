---
title: "RepCali: High Efficient Fine-tuning Via Representation Calibration in Latent Space for Pre-trained Language Models"
pubDatetime: 2025-05-13T11:47:00+00:00
slug: "2025-05-representation-calibration"
type: "arxiv"
id: "2505.08463"
score: 0.8234080266164495
author: "grok-3-latest"
authors: ["Fujun Zhang", "XiangDong Su"]
tags: ["LLM", "Pre-Training", "Post-Training", "Latent Space", "Fine-Tuning"]
institution: ["Inner Mongolia University"]
description: "本文提出 RepCali 方法，通过在潜在空间校准编码器输出的表征，以极低参数成本显著提升预训练语言模型在下游任务中的性能。"
---

> **Summary:** 本文提出 RepCali 方法，通过在潜在空间校准编码器输出的表征，以极低参数成本显著提升预训练语言模型在下游任务中的性能。 

> **Keywords:** LLM, Pre-Training, Post-Training, Latent Space, Fine-Tuning

**Authors:** Fujun Zhang, XiangDong Su

**Institution(s):** Inner Mongolia University


## Problem Background

预训练语言模型（PLMs）在下游任务中因领域差异和目标差异而面临性能瓶颈，具体表现为编码器输出的表征与解码器期望的最优输入分布之间存在显著不匹配。
论文旨在通过在潜在空间中校准表征，缩小这种差异，从而提升模型在下游任务中的表现。

## Method

*   **核心思想:** 在编码器和解码器之间引入一个校准模块（Calibration Block），通过学习调整编码器输出的表征，使其更接近解码器所需的最优分布。
*   **具体实现:** 
    *   初始化一个形状种子矩阵（Shape Seed），尺寸与输入匹配，初始值为全1矩阵，作为校准的基础输入。
    *   使用一个可学习的嵌入层（Learnable Embedding）对 Shape Seed 进行编码，生成校准值，用于调整编码器输出。
    *   将校准值与编码器输出相加，并通过层归一化（Layer Normalization）处理，确保校准后的表征稳定且有效。
    *   引入超参数 λ 控制校准程度，避免过度干扰原始表征，保持模型的稳定性。
*   **特点:** 该方法即插即用（plug-and-play），适用于所有编码器-解码器架构的 PLMs，参数增加量极小（0-0.8%），实现简单且高效。

## Experiment

*   **有效性:** RepCali 在 25 个 PLM 模型和 8 个下游任务（涵盖英文和中文数据集）中均显著提升性能，例如在 SST2、RET、MNLI 和 CoLA 数据集上，相较 LoRA 和 Adapter 提升超过 1%；在 MultiWOZ 数据集上，MinTL (BART-large) 的 Inform 和 Success 指标分别提升 4.11% 和 5.37%。
*   **效率:** 仅增加 0-0.8% 的参数量，远低于其他微调方法（如 Adapter 和 Prefix-Tuning），体现了高效性。
*   **全面性与合理性:** 实验覆盖多种任务类型（生成与理解任务）和语言，数据集选择具有代表性（如 XSum、WebNLG、CSDS），结果基于 3 个随机种子平均值，可信度高；潜在空间可视化（t-SNE）证实校准后表征分布更紧凑有序。
*   **局限性:** 部分任务指标提升幅度较小（如 XSum 数据集 ROUGE-1 仅提升 0.11%），可能与任务特性或模型规模有关；未深入探讨超参数 λ 的最优值选择。

## Further Thoughts

RepCali 的潜在空间校准思想启发了对表征优化的进一步探索：是否可以结合任务特定先验知识（如领域或角色信息）设计针对性校准模块？是否能与 LoRA 或 Adapter 等方法结合形成混合策略？此外，是否可在预训练阶段引入类似校准机制，减少后续微调的领域适应成本？