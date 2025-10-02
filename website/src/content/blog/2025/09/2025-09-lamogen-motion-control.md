---
title: "LaMoGen: Laban Movement-Guided Diffusion for Text-to-Motion Generation"
pubDatetime: 2025-09-29T08:48:49+00:00
slug: "2025-09-lamogen-motion-control"
type: "arxiv"
id: "2509.24469"
score: 0.7252517788843221
author: "grok-3-latest"
authors: ["Heechang Kim", "Gwanghyun Kim", "Se Young Chun"]
tags: ["Diffusion Model", "Text-to-Motion", "Motion Control", "Expressive Generation"]
institution: ["Seoul National University"]
description: "本文提出 LaMoGen 框架，通过拉班运动分析引导预训练扩散模型，在推理时实现文本到动作生成的细粒度表现力控制，无需额外训练数据。"
---

> **Summary:** 本文提出 LaMoGen 框架，通过拉班运动分析引导预训练扩散模型，在推理时实现文本到动作生成的细粒度表现力控制，无需额外训练数据。 

> **Keywords:** Diffusion Model, Text-to-Motion, Motion Control, Expressive Generation

**Authors:** Heechang Kim, Gwanghyun Kim, Se Young Chun

**Institution(s):** Seoul National University


## Problem Background

文本到动作生成（Text-to-Motion Generation）是计算机视觉和人机交互领域的重要任务，但现有基于扩散模型的方法难以通过文本描述实现对动作风格和表现力的细粒度控制。
这一问题源于动作数据集规模较小，难以捕捉细微表现力差异，以及自然语言在描述非语言属性（如能量、流畅性）时的模糊性和量化难度。

## Method

*   **核心思想:** 提出 LaMoGen 框架，通过拉班运动分析（Laban Movement Analysis, LMA）的 Effort 和 Shape 组件，结合推理时优化策略，引导预训练扩散模型生成具有特定表现力的动作，而无需额外训练数据。
*   **具体实现:** 
    *   **LMA 特征量化:** 将 LMA 的四个组件（Weight、Time、Flow、Shape）定义为可微分的时间序列特征（如基于速度、加速度、抖动等计算），通过整体约束间接控制峰值，确保适用于梯度优化。
    *   **两步生成策略:** 第一步，基于纯文本提示生成基准动作，提取其 LMA 特征，并根据用户指定的标签（如‘strong’或‘sudden’）或直接缩放向量定义目标特征；第二步，从相同初始噪声开始，通过 DDIM 采样过程引导生成最终动作。
    *   **推理时引导机制:** 在每个采样步骤中，计算生成的中间动作与目标 LMA 特征之间的‘Laban 损失’，通过梯度下降优化文本嵌入，使生成的动作特征逐步接近目标，同时保持动作内容一致性。
*   **关键优势:** 方法为零样本，仅在推理阶段操作，适用于现有扩散模型（如 MotionDiffuse），无需重新训练或标注数据，具有高灵活性。

## Experiment

*   **有效性:** 在 HumanML3D 数据集上的定性结果显示，LaMoGen 能生成符合目标 LMA 标签的动作（如‘Strong’标签下动作更具力量感），显著优于提示词编辑等基线方法，后者往往无法产生明显风格变化。
*   **定量提升:** 在可控性与解耦性方面，LaMoGen 取得最高 Diagonality 分数（0.978），表明其能针对特定 LMA 组件进行修改，而对其他组件影响最小；相较无引导基线，R-Precision 和 FID 略有下降（如 R-Precision Top-1 从 0.470 降至 0.424），但仍优于其他控制方法，体现了合理的性能-控制权衡。
*   **实验设置合理性:** 实验对比了多种基线（提示词编辑、原始帧更新、分类器引导），并通过消融研究验证了高斯平滑和学习率等设计的必要性，设置全面且结果可信。

## Further Thoughts

将艺术领域的结构化分析框架（如 LMA）与深度生成模型结合，为生成任务引入可解释性和细粒度控制，这种跨领域思路可推广至其他领域（如音乐或图像生成）；推理时优化策略无需额外训练数据，是否能将其他外部约束（如物理规则）动态融入生成过程？此外，LMA 特征解耦控制的思路启发我们，未来可通过用户研究或强化学习优化特征映射，使其更符合人类感知，而非依赖启发式设计。