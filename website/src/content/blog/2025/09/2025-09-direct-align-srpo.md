---
title: "Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference"
pubDatetime: 2025-09-08T17:54:08+00:00
slug: "2025-09-direct-align-srpo"
type: "arxiv"
id: "2509.06942"
score: 0.5028937998253464
author: "grok-3-latest"
authors: ["Xiangwei Shen", "Zhimin Li", "Zhantao Yang", "Shiyi Zhang", "Yingfang Zhang", "Donghao Li", "Chunyu Wang", "Qinglin Lu", "Yansong Tang"]
tags: ["Diffusion Models", "Human Preference", "Reward Optimization", "Online RL", "Text Conditioning"]
institution: ["Hunyuan, Tencent", "The Chinese University of Hong Kong, Shenzhen", "Shenzhen International Graduate School, Tsinghua University"]
description: "本文提出 Direct-Align 和 SRPO 框架，通过单步图像恢复和语义相对偏好优化，解决了扩散模型在全轨迹优化和细粒度人类偏好对齐中的关键问题，显著提升了生成图像真实感和审美质量，并大幅提高训练效率。"
---

> **Summary:** 本文提出 Direct-Align 和 SRPO 框架，通过单步图像恢复和语义相对偏好优化，解决了扩散模型在全轨迹优化和细粒度人类偏好对齐中的关键问题，显著提升了生成图像真实感和审美质量，并大幅提高训练效率。 

> **Keywords:** Diffusion Models, Human Preference, Reward Optimization, Online RL, Text Conditioning

**Authors:** Xiangwei Shen, Zhimin Li, Zhantao Yang, Shiyi Zhang, Yingfang Zhang, Donghao Li, Chunyu Wang, Qinglin Lu, Yansong Tang

**Institution(s):** Hunyuan, Tencent, The Chinese University of Hong Kong, Shenzhen, Shenzhen International Graduate School, Tsinghua University


## Problem Background

扩散模型在与人类偏好对齐时面临两大挑战：一是现有方法依赖多步去噪计算奖励梯度，计算成本高，导致优化仅限于后期时间步，易引发奖励黑客问题（即生成高奖励但低质量图像）；二是缺乏在线调整奖励模型的机制，需离线预调以适应特定审美需求，增加了复杂性。
本文旨在解决如何在全扩散轨迹上高效优化并动态调整奖励信号，以实现细粒度的人类偏好对齐。

## Method

*   **Direct-Align**：针对多步去噪计算成本高的问题，提出了一种单步图像恢复策略，通过预定义噪声先验注入图像，从任意时间步直接插值恢复原始图像，避免后期时间步过优化，支持早期时间步优化，减少梯度爆炸等不稳定因素。
*   **Semantic Relative Preference Optimization (SRPO)**：针对奖励模型调整问题，将奖励信号视为文本条件信号，通过正向和负向提示词增强在线调整奖励，减少对离线奖励微调的依赖；同时引入语义相对偏好机制，利用正负提示对的相对差异作为目标函数，过滤无关信息，缓解奖励黑客问题。
*   **实现细节**：在训练中，Direct-Align 通过噪声注入和单步去噪/反转恢复图像，结合奖励聚合框架（使用衰减折扣因子）进一步稳定优化；SRPO 则通过文本嵌入操控奖励特性，支持去噪和反转双向优化，确保优化方向与目标偏好一致。

## Experiment

*   **有效性**：基于 FLUX.1.dev 模型，在 HPDv2 数据集上，SRPO 方法在人类评估中显著提升，真实感优秀率从 8.2% 提升至 38.9%，审美质量优秀率从 9.8% 提升至 40.5%，整体偏好优秀率从 5.3% 提升至 29.4%，相比基线提升约 3-4 倍。
*   **优越性**：与现有在线强化学习方法（如 ReFL, DRaFT, DanceGRPO）相比，SRPO 在多个自动评估指标（如 Aesthetic Score, PickScore）和人类评估中表现更优，且对奖励黑客问题有更好免疫性。
*   **效率**：训练效率极高，仅需 10 分钟（使用 32 个 NVIDIA H20 GPU）即可收敛，相比 DanceGRPO 提升了 75 倍。
*   **实验设置合理性**：实验涵盖自动指标和人类评估，人类评估涉及 10 名训练过的标注者和 3 名领域专家，评估维度包括真实感、细节复杂性、审美质量等，确保结果可信；但也存在局限，如对奖励模型感知能力较弱的控制词效果不佳。

## Further Thoughts

文本条件奖励信号的在线调整机制启发了我，是否可以将这一思路扩展到其他生成任务（如视频或语音合成），通过动态提示词操控生成方向？此外，Direct-Align 的单步恢复策略为解决多步计算瓶颈提供了新思路，或许可以结合其他高效采样方法进一步提升性能；相对偏好机制也可能适用于其他强化学习场景，帮助缓解奖励函数设计中的主观性问题。