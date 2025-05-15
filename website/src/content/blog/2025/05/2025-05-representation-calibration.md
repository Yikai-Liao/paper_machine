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
description: "本文提出 RepCali 方法，通过在潜在空间中校准编码器输出的表征分布，显著提升预训练语言模型在下游任务中的性能，同时保持极低的参数开销。"
---

> **Summary:** 本文提出 RepCali 方法，通过在潜在空间中校准编码器输出的表征分布，显著提升预训练语言模型在下游任务中的性能，同时保持极低的参数开销。 

> **Keywords:** LLM, Pre-Training, Post-Training, Latent Space, Fine-Tuning

**Authors:** Fujun Zhang, XiangDong Su

**Institution(s):** Inner Mongolia University


## Problem Background

预训练语言模型（PLMs）在下游任务中因领域差异和目标差异而表现受限，传统微调方法虽有改善，但编码器输出的表征与解码器期望的最优输入之间仍存在显著分布差异，导致性能瓶颈。

## Method

* **核心思想**：通过在编码器和解码器之间的潜在空间中引入一个校准模块（Calibration Block），调整编码器输出的表征分布，使其更接近解码器的最优输入需求，从而提升下游任务性能。
* **具体实现**：
  - **Shape Seed**：初始化一个与输入维度匹配的全1矩阵，作为校准的起点。
  - **Learnable Embedding**：通过一个可学习的嵌入层对 Shape Seed 进行编码，生成校准值，用于调整编码器输出。
  - **校准过程**：将校准值与编码器输出在潜在空间中相加，并通过层归一化（Layer Normalization）处理，生成校准后的表征作为解码器输入。
  - **超参数控制**：引入超参数 λ 控制校准程度，确保校准效果与模型性能的平衡。
* **特点**：方法具有即插即用（plug-and-play）特性，适用于所有编码器-解码器架构的 PLMs，仅增加少量参数（0-0.8%），不需修改原始模型结构或损失函数。
* **与其他方法的区别**：不同于 Prompt Tuning（通过输入嵌入添加提示）、Adapter（在层间插入模块）或 LoRA（低秩自注意力调整），RepCali 直接作用于潜在空间的表征校准，聚焦于编码器-解码器之间的分布匹配。

## Experiment

* **性能提升**：RepCali 在 25 个 PLM 模型和 8 个下游任务（包括英文和中文数据集）上均取得显著改进，例如在 SST2、RET、MNLI 和 CoLA 数据集上，相较 LoRA 和 Adapter 提升超过 1%；在 MultiWOZ 数据集上，MinTL (BART-large) 的 Inform 和 Success 指标分别提升 4.11% 和 5.37%。
* **参数效率**：仅增加 0-0.8% 的额外参数，远低于其他微调方法（如 Adapter 增加 2.38%），体现出高效性。
* **实验设置**：实验覆盖多种模型规模（T5-small 到 T5-3B）和任务类型（生成与理解任务），结果基于 3 个随机种子的平均值，设置全面合理，验证了方法的通用性和跨语言适应性。
* **可视化验证**：通过 t-SNE 可视化潜在空间，RepCali 使表征分布更紧凑有序，证实了校准效果。
* **局限性**：未深入探讨超参数 λ 的任务特异性调节策略及校准模块对不同任务的适应性差异。

## Further Thoughts

RepCali 的潜在空间校准理念可扩展至多模态模型（如视觉-语言模型），通过校准图像编码器与文本解码器之间的表征分布提升性能；此外，是否可结合知识蒸馏，通过校准教师与学生模型的表征差异提高蒸馏效率；或者引入动态校准机制，根据任务特性调整校准策略，进一步优化效果。