---
title: "Fast Text-to-Audio Generation with Adversarial Post-Training"
pubDatetime: 2025-05-13T02:25:47+00:00
slug: "2025-05-arc-text-audio"
type: "arxiv"
id: "2505.08175"
score: 0.49297930386133737
author: "grok-3-latest"
authors: ["Zachary Novack", "Zach Evans", "Zack Zukowski", "Josiah Taylor", "CJ Carr", "Julian Parker", "Adnan Al-Sinan", "Gian Marco Iodice", "Julian McAuley", "Taylor Berg-Kirkpatrick", "Jordi Pons"]
tags: ["Text-to-Audio", "Generative Model", "Adversarial Training", "Sampling", "Post-Training"]
institution: ["UC San Diego", "Stability AI", "Arm"]
description: "本文提出 ARC 后训练方法，通过对抗性相对论损失和对比损失加速文本到音频生成，在保持质量和提升多样性的同时，将推理时间缩短至毫秒级（H100 GPU）或秒级（边缘设备）。"
---

> **Summary:** 本文提出 ARC 后训练方法，通过对抗性相对论损失和对比损失加速文本到音频生成，在保持质量和提升多样性的同时，将推理时间缩短至毫秒级（H100 GPU）或秒级（边缘设备）。 

> **Keywords:** Text-to-Audio, Generative Model, Adversarial Training, Sampling, Post-Training

**Authors:** Zachary Novack, Zach Evans, Zack Zukowski, Josiah Taylor, CJ Carr, Julian Parker, Adnan Al-Sinan, Gian Marco Iodice, Julian McAuley, Taylor Berg-Kirkpatrick, Jordi Pons

**Institution(s):** UC San Diego, Stability AI, Arm


## Problem Background

文本到音频（Text-to-Audio）生成模型尽管性能不断提升，但在推理时速度较慢，通常需要几秒到几分钟生成一段音频，这限制了其在创意应用（如音乐制作、音效设计）中的实用性。
论文旨在加速基于高斯流的生成模型（如扩散模型或整流流模型），解决推理速度与生成质量之间的矛盾，同时避免传统蒸馏方法的高训练成本和生成多样性下降的问题。

## Method

*   **核心思想**：提出 Adversarial Relativistic-Contrastive (ARC) 后训练方法，通过对抗性训练将预训练的整流流模型转化为少步（1-8步）生成器，显著加速文本到音频生成，同时尽量保持质量和多样性。
*   **具体实现**：
    *   **对抗性相对论损失（Adversarial Relativistic Loss）**：基于相对论 GAN 思想，生成器和判别器基于成对的真实/生成样本进行相对优化，生成器试图让生成样本在判别器空间中‘比真实样本更真实’，而判别器则相反；由于样本对共享相同文本提示，提供了更强的梯度信号。
    *   **对比损失（Contrastive Loss）**：为解决对抗性训练中提示遵循性不足的问题，引入对比损失，训练判别器区分正确和错误（随机打乱）的音频-文本对，增强判别器对语义特征的关注，间接提升生成器的提示遵循性。
    *   **Ping-Pong 采样**：推理时采用 Ping-Pong 采样策略，通过交替去噪和重新加噪逐步优化生成样本，避免传统 ODE 求解器所需的大量步骤，同时不使用分类器自由引导（CFG），减少内存开销。
    *   **架构优化**：基于 Stable Audio Open (SAO) 模型，优化 Diffusion Transformer (DiT) 架构（减少层数和维度），并结合编译技术进一步加速推理。
*   **关键优势**：ARC 是首个完全基于对抗性后训练的文本到音频加速框架，不依赖蒸馏或 CFG，降低了训练成本（无需存储多个模型或大量轨迹数据），并试图通过对比损失平衡提示遵循性和多样性。

## Experiment

*   **速度提升**：ARC 后训练使模型在 H100 GPU 上生成约 12 秒的 44.1kHz 立体声音频仅需 75 毫秒，比原始 SAO 模型快 100 倍；在移动边缘设备上生成时间为约 7 秒，是目前已知最快的文本到音频模型；实时因子（RTF）高达 440.30（4步模型），速度优势显著。
*   **质量与多样性**：在客观指标（如 FD_openl3, KL_passt, CLAP 评分）上，ARC 与蒸馏方法（如 Presto）和其他基线（如 SAO）相比具有竞争力，尽管在某些质量指标（如 MOS 质量评分）略低于 Presto；但在多样性上表现突出，尤其是在主观 MOS 评分和提出的 CCDS（CLAP Conditional Diversity Score）指标上，显著优于 Presto，适合创意应用。
*   **实验设置**：实验涵盖多种步骤配置（1步、4步、8步）、不同设备（H100 GPU、移动设备）、多种评估指标（质量、提示遵循性、多样性、速度）以及主观听觉测试；数据量充足（如 AudioCaps 测试集生成 4875 个音频样本），对比基线合理；不足之处在于主观测试参与者数量较少（14 人），可能影响统计显著性。
*   **消融研究**：单独使用相对论损失（无对比损失）会导致提示遵循性下降，但多样性增加；相对论损失优于传统最小二乘对抗损失，验证了 ARC 设计的合理性。

## Further Thoughts

ARC 的对抗性后训练替代蒸馏的思路可能启发其他模态（如图像、视频生成）探索低成本加速策略；相对论与对比损失结合的方法可应用于其他条件生成任务以平衡提示遵循性和多样性；提出的 CCDS 多样性评估指标与主观测试高度一致，或将成为生成模型评估新标准；边缘设备优化（如动态 Int8 量化）表明生成模型在资源受限环境下的部署潜力，可能推动其在消费级设备上的普及应用。