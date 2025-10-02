---
title: "UniLat3D: Geometry-Appearance Unified Latents for Single-Stage 3D Generation"
pubDatetime: 2025-09-29T17:21:23+00:00
slug: "2025-09-unilat3d-single-stage"
type: "arxiv"
id: "2509.25079"
score: 0.91951552037091
author: "grok-3-latest"
authors: ["Guanjun Wu", "Jiemin Fang", "Chen Yang", "Sikuang Li", "Taoran Yi", "Jia Lu", "Zanwei Zhou", "Jiazhong Cen", "Lingxi Xie", "Xiaopeng Zhang", "Wei Wei", "Wenyu Liu", "Xinggang Wang", "Qi Tian"]
tags: ["3D Generation", "Unified Representation", "Latent Space", "Single-Stage Generation", "Geometry Appearance"]
institution: ["Huawei Inc.", "Huazhong University of Science and Technology", "Shanghai Jiaotong University"]
description: "UniLat3D 提出几何-外观统一的潜在表示和单阶段生成框架，显著提升了3D资产生成的质量与效率，为3D生成领域提供了简洁且可扩展的新范式。"
---

> **Summary:** UniLat3D 提出几何-外观统一的潜在表示和单阶段生成框架，显著提升了3D资产生成的质量与效率，为3D生成领域提供了简洁且可扩展的新范式。 

> **Keywords:** 3D Generation, Unified Representation, Latent Space, Single-Stage Generation, Geometry Appearance

**Authors:** Guanjun Wu, Jiemin Fang, Chen Yang, Sikuang Li, Taoran Yi, Jia Lu, Zanwei Zhou, Jiazhong Cen, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Xinggang Wang, Qi Tian

**Institution(s):** Huawei Inc., Huazhong University of Science and Technology, Shanghai Jiaotong University


## Problem Background

当前高质量3D资产生成多采用基于扩散模型的两阶段流程，先生成几何形状（Geometry），再合成外观纹理（Appearance），这种分离设计导致几何与纹理不一致（misalignment）并增加计算成本。
作者受计算机视觉领域从多阶段到单阶段演进的启发，提出一个统一的框架，将几何和外观编码到单一潜在空间，实现单阶段生成，以解决不一致性和效率问题。

## Method

*   **核心思想：** 提出 UniLat3D 框架，通过一个几何-外观统一的潜在表示（UniLat），将3D资产的几何和外观信息融合到一个紧凑的潜在空间中，实现单阶段高效生成。
*   **统一潜在表示（UniLat）：** 将高分辨率稀疏特征压缩为低分辨率密集潜在表示，融合几何和外观信息，避免两阶段分离带来的不一致性。
*   **Uni-VAE 结构：** 设计一个统一的变分自编码器（Uni-VAE），包括编码器和解码器两部分：
    *   **编码器：** 包含稀疏视觉特征提取、稀疏外观编码、稀疏特征致密化和致密特征压缩四个阶段，通过稀疏Transformer和3D卷积层将3D资产编码为UniLat。
    *   **解码器：** 包含上采样模块和3D表示解码模块，通过上采样和稀疏化恢复高分辨率特征，支持输出多种3D格式（如3D高斯和网格），并通过分层监督和细节增强策略提升网格分辨率至512³。
*   **单阶段生成：** 基于流匹配模型（Flow-Matching Model），训练一个统一的流Transformer（F_uni），直接从高斯噪声映射到UniLat，省去中间几何和外观分离生成的步骤。
*   **训练与优化：** 使用公开数据集（如Objaverse、ABO）训练，结合多种损失函数（如L1、LPIPS、SSIM、KL散度、Dice损失）优化几何和外观重建质量，同时采用两阶段训练策略降低高分辨率网格生成成本。

## Experiment

*   **有效性：** 在 Toys4K 和自建复杂数据集上，UniLat3D 在外观保真度（CLIP 得分高达90.87）和几何质量（ULIP 得分42.69）上显著优于大多数对比方法（如 TRELLIS、Hunyuan3D-2.1），生成的3D资产与条件图像一致性更高。
*   **效率：** 3D高斯生成仅需8秒（A100 GPU），网格生成需36秒，虽比 TRELLIS 稍慢，但考虑到更高的分辨率（网格分辨率达512³）和质量，效率仍具竞争力。
*   **实验设置：** 数据集选择合理，涵盖公开数据集和自建复杂数据集，评价指标全面（包括 PSNR、SSIM、LPIPS、CLIP、FD DINOv2 等），覆盖重建和生成质量；用户研究显示 UniLat3D 在图像对齐和对象质量上获得超过35%的投票，优于其他模型。
*   **消融研究：** 潜在空间分辨率越高（如32³），重建质量越好；使用 DINOv3 作为条件图像编码器相较 DINOv2 进一步降低 FD DINOv2 至49.90，表明更好的编码器能提升复杂对象生成质量。

## Further Thoughts

UniLat 作为统一潜在表示的创新，不仅限于3D资产生成，还可能扩展至4D表示（加入时间维度）或场景生成；其作为3D先验（Prior）集成到大型多模态模型中的潜力，为跨模态任务（如文本-3D、图像-3D）提供了新思路；此外，单阶段生成的思想启发我们在其他生成任务中探索统一表示，减少多阶段误差累积。