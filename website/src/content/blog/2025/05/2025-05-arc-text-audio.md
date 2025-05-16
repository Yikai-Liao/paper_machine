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
description: "本文提出 Adversarial Relativistic-Contrastive (ARC) 后训练方法，通过结合相对论对抗损失和对比损失，显著加速文本到音频生成模型，同时保持生成多样性，为创意应用提供实用性。"
---

> **Summary:** 本文提出 Adversarial Relativistic-Contrastive (ARC) 后训练方法，通过结合相对论对抗损失和对比损失，显著加速文本到音频生成模型，同时保持生成多样性，为创意应用提供实用性。 

> **Keywords:** Text-to-Audio, Generative Model, Adversarial Training, Sampling, Post-Training

**Authors:** Zachary Novack, Zach Evans, Zack Zukowski, Josiah Taylor, CJ Carr, Julian Parker, Adnan Al-Sinan, Gian Marco Iodice, Julian McAuley, Taylor Berg-Kirkpatrick, Jordi Pons

**Institution(s):** UC San Diego, Stability AI, Arm


## Problem Background

文本到音频（Text-to-Audio）生成系统尽管性能不断提升，但在推理时速度较慢，通常需要几秒到几分钟生成音频，这在创意应用场景（如音乐制作或实时音效生成）中限制了其实用性。
论文旨在解决高斯流模型（如扩散模型和整流流模型）推理时间过长的问题，同时尽量保持生成质量和多样性，克服传统蒸馏方法带来的高训练成本、存储需求以及多样性下降等缺陷。

## Method

*   **核心思想:** 提出一种名为 Adversarial Relativistic-Contrastive (ARC) 的后训练（Post-Training）方法，通过对抗性训练将预训练的高斯流模型转化为少步采样（1-8步）的生成器，从而显著加速文本到音频生成，同时避免使用蒸馏和 Classifier-Free Guidance (CFG)。
*   **具体实现:** 
    *   **整流流预训练:** 基于整流流（Rectified Flow）模型，学习从噪声到数据的转换过程，为后续加速提供基础模型。
    *   **对抗性相对论损失 (Adversarial Relativistic Loss):** 设计生成器和判别器之间的对抗训练，基于成对的真实和生成样本（共享相同文本提示）计算相对‘真实性’，使生成器试图生成比真实样本‘更真实’的输出，判别器则试图区分真实样本的相对优势，提供更强的梯度信号以提升生成质量。
    *   **对比损失 (Contrastive Loss):** 训练判别器区分正确和错误的音频-文本对，通过最大化正确和错误提示下判别器输出的差异，增强模型对文本提示的遵循能力，弥补对抗训练中提示遵循性不足的问题。
    *   **乒乓采样 (Ping-Pong Sampling):** 采用一种迭代去噪和再加噪的采样策略，从初始噪声样本开始，通过多次去噪和降低噪声级别的再加噪逐步优化生成结果，避免传统 ODE 求解器需要大量步骤的问题，同时不使用内存密集的 CFG。
    *   **架构优化:** 对 Stable Audio Open (SAO) 模型进行改进，减少 Diffusion Transformer (DiT) 参数量（从 1.06B 降至 0.34B），添加 QK-LayerNorm 等优化，并通过编译技术提升推理效率。
*   **关键优势:** ARC 不依赖蒸馏，避免了生成和存储轨迹输出对的高成本，也无需在内存中同时存储多个模型；通过对抗性训练直接利用真实数据而非教师模型生成样本，摆脱了对预训练模型性能的依赖。

## Experiment

*   **速度提升:** ARC 方法在 H100 GPU 上生成约 12 秒的 44.1kHz 立体声音频仅需 75ms，相比原始 Stable Audio Open (SAO) 模型快 100 倍；在移动边缘设备（如 Vivo X200 Pro）上生成时间为 6.6 秒，实时因子 (RTF) 最高达到 440.30（4步采样），显著优于 SAO 的 3.56。
*   **生成质量与提示遵循性:** 在客观指标（如 FDopenl3, KLpasst, CLAP score）上，ARC 与其他加速方法（如 Presto）相比具有竞争力，但略低于未加速的 SAO 模型；主观评估 (MOS) 显示 ARC 在质量和提示遵循性上稍逊于 Presto，但仍保持在合理范围内。
*   **生成多样性:** ARC 在多样性指标（CCDS, Rpasst, Cpasst）以及主观多样性评分上显著优于 Presto，尤其在避免 CFG 带来的过饱和和低多样性问题上表现突出。
*   **消融实验验证:** 去掉对比损失 (LC) 后，提示遵循性显著下降；使用传统最小二乘对抗损失替代相对论损失后，整体性能不如 ARC，证明了 ARC 双损失设计的有效性。
*   **实验设置合理性:** 实验覆盖多种基线模型（SAO, Presto, Pre-trained RF），评估指标全面（质量、提示遵循性、多样性、速度），数据集（AudioCaps）选择合理，主观测试针对创意应用场景设计特定提示，整体设置较为全面，但质量与最优基线仍有差距，体现了加速与质量之间的权衡。

## Further Thoughts

1. **对抗性后训练替代蒸馏的潜力:** ARC 展示了非蒸馏方法在加速生成模型中的可能性，这可能启发其他领域（如图像或视频生成）探索类似的对抗性后训练策略，以避免蒸馏带来的高成本和多样性损失。
2. **相对论与对比损失的结合应用:** 通过相对论损失增强对抗效果，并用对比损失提升语义一致性，这种双损失设计可能适用于其他条件生成任务，以解决对抗训练中条件遵循性不足的问题。
3. **边缘设备优化的实用性:** 论文在移动设备上的优化（如动态 Int8 量化）表明生成模型在资源受限环境下的应用潜力，可能推动更多实时创意工具的开发。
4. **多样性评估新指标 (CCDS):** 提出的 CLAP Conditional Diversity Score 为条件生成多样性提供了一种自动化评估方法，这在生成模型评估中是一个有意义的创新，可能被广泛应用于其他生成任务。