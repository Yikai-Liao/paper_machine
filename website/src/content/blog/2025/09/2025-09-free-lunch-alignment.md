---
title: "Free Lunch Alignment of Text-to-Image Diffusion Models without Preference Image Pairs"
pubDatetime: 2025-09-30T04:32:34+00:00
slug: "2025-09-free-lunch-alignment"
type: "arxiv"
id: "2509.25771"
score: 0.76873054816945
author: "grok-3-latest"
authors: ["Jia Jun Cheng Xian", "Muchen Li", "Haotian Yang", "Xin Tao", "Pengfei Wan", "Leonid Sigal", "Renjie Liao"]
tags: ["LLM", "Text-to-Image", "Diffusion Model", "Preference Optimization", "Alignment"]
institution: ["University of British Columbia", "Vector Institute for AI", "Canada CIFAR AI Chair", "Kling Team at Kuaishou Technology", "NSERC CRC Chair"]
description: "本文提出了一种无需人类偏好图像对的‘免费午餐’对齐框架，通过LLM生成的文本偏好对优化文本到图像扩散模型的对齐性能，并在多个基准测试中取得优于传统方法的结果。"
---

> **Summary:** 本文提出了一种无需人类偏好图像对的‘免费午餐’对齐框架，通过LLM生成的文本偏好对优化文本到图像扩散模型的对齐性能，并在多个基准测试中取得优于传统方法的结果。 

> **Keywords:** LLM, Text-to-Image, Diffusion Model, Preference Optimization, Alignment

**Authors:** Jia Jun Cheng Xian, Muchen Li, Haotian Yang, Xin Tao, Pengfei Wan, Leonid Sigal, Renjie Liao

**Institution(s):** University of British Columbia, Vector Institute for AI, Canada CIFAR AI Chair, Kling Team at Kuaishou Technology, NSERC CRC Chair


## Problem Background

文本到图像（Text-to-Image, T2I）扩散模型在生成高质量图像方面表现出色，但文本与图像的对齐（alignment）问题仍是关键挑战。
传统方法依赖于强化学习结合人类反馈（RLHF），需要大量昂贵的人类标注偏好数据（如成对图像偏好），存在成本高和扩展性差的问题。
本文旨在探索一种无需人类偏好图像对的‘免费午餐’方法，通过利用现有高质量图像-文本数据集和文本偏好优化来提升T2I模型的对齐性能。

## Method

*   **核心思想:** 提出Text Preference Optimization (TPO)框架，通过构建匹配和不匹配的文本提示对（而非图像对），优化T2I扩散模型的文本-图像对齐，避免对人类偏好数据的依赖。
*   **文本偏好对构建:** 利用大型语言模型（LLM）对原始图像标题进行扰动，生成不匹配提示（negative prompts），基于四种修改原则（内容修改、属性修改、空间修改、上下文修改）确保语义差异。例如，将‘inside’改为‘outside’以改变空间布局。
*   **优化目标设计:** 基于Direct Preference Optimization (DPO)和Kahneman-Tversky Optimization (KTO)，提出两种变体TDPO和TKTO，通过对比匹配和不匹配提示对的扩散损失，优化模型以更倾向于生成与匹配提示一致的图像。具体而言，TDPO使用Bradley-Terry模型建模偏好关系，TKTO基于前景理论设计效用函数，两者均通过闭式优化避免显式奖励建模。
*   **扩散模型适配:** 将TPO应用于扩散模型（如Stable Diffusion v1.5），通过调整训练损失（如引入剪切机制以稳定训练，限制负样本梯度带来的不稳定性），确保模型在不牺牲生成质量的前提下提升对齐性能。
*   **关键优势:** 方法通用，可无缝集成到现有基于偏好的RLHF框架中，且通过‘免费午餐’理念大幅降低标注成本。

## Experiment

*   **有效性:** 实验基于Stable Diffusion v1.5，在多个数据集（如HPDv2、Pick-a-Pic v2、Parti-Prompts）上评估，TDPO和TKTO在大多数指标（如PickScore、CLIP alignment、HPSv2、ImageReward）上显著优于基线方法（如Diffusion-DPO和Diffusion-KTO）。例如，在HPSv2数据集上，TDPO的PickScore胜率为83.25%，远高于Diffusion-DPO的77.00%。
*   **合理性:** 实验设置全面，涵盖多个数据集和评估指标，通过消融研究验证了提示修改原则的有效性（如内容修改对CLIP分数影响显著）以及不同修改预算的影响，同时定性结果显示方法在捕捉提示细节（如‘twilight’和‘misty’）方面更优。
*   **局限性与开销:** 尽管性能提升明显，但方法对提示编辑质量依赖较高，固定文本编码器可能限制细微语义区分能力；计算开销主要来自LLM生成负样本和额外训练步骤，但整体成本远低于人类标注。

## Further Thoughts

文本偏好替代图像偏好的思路非常具有启发性，未来可以推广到其他多模态任务（如文本到视频或3D生成），探索更复杂的语义扰动策略以生成挑战性负样本；此外，利用LLM作为数据增强工具的理念可以结合对抗性生成或多模型协作，进一步提升负样本多样性，解决标注数据稀缺问题；最后，隐式偏好与人类偏好相关性的发现为设计高效偏好优化算法提供了理论支持，或许可以通过动态调整文本编码器来增强对细微语义的敏感性。