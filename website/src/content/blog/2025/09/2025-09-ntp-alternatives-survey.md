---
title: "Alternatives To Next Token Prediction In Text Generation -- A Survey"
pubDatetime: 2025-09-29T08:18:16+00:00
slug: "2025-09-ntp-alternatives-survey"
type: "arxiv"
id: "2509.24435"
score: 0.7797484846792172
author: "grok-3-latest"
authors: ["Charlie Wyatt", "Aditya Joshi", "Flora Salim"]
tags: ["LLM", "Text Generation", "Planning", "Latent Space", "Diffusion Model"]
institution: ["UNSW Sydney, School of Computer Science and Engineering"]
description: "本文通过系统性综述，将 Next Token Prediction 的替代方案分类为五大类，首次提供统一框架，梳理研究现状并指引未来方向。"
---

> **Summary:** 本文通过系统性综述，将 Next Token Prediction 的替代方案分类为五大类，首次提供统一框架，梳理研究现状并指引未来方向。 

> **Keywords:** LLM, Text Generation, Planning, Latent Space, Diffusion Model

**Authors:** Charlie Wyatt, Aditya Joshi, Flora Salim

**Institution(s):** UNSW Sydney, School of Computer Science and Engineering


## Problem Background

大型语言模型（LLMs）依赖的 Next Token Prediction (NTP) 范式虽然推动了文本生成的成功，但存在显著局限性，包括贪婪生成导致的长文本连贯性不足、token 级粒度与人类语义思维的不匹配、计算效率低下（Transformer 的二次方复杂度）以及训练时监督错位问题；论文旨在通过系统性综述，探索和分类 NTP 的替代方案，为解决这些关键问题提供研究路线图。

## Method

*   **综述目标与方法**：论文通过系统性文献综述，梳理 2020-2025 年间主要 NLP 会议和预印本中的研究，将 NTP 替代方案分类为五大类，分析其定义、代表性工作、失败模式及潜力。
*   **分类框架**：
    *   **Multi-Token Prediction (MTP)**：改变预测目标，从单个 token 到多个未来 token，使用共享上下文表示（trunk）并行预测，以提升短程规划和计算效率；如 ProphetNet 使用 n-stream 自注意力机制，Gloeckle et al. 在训练时引入多输出头。
    *   **Plan-then-Generate (PtG)**：引入两阶段过程，先生成全局计划（象征性或潜在表示），再基于计划指导 token 生成，增强长程连贯性；如 PlanGen 使用 CRF 预测内容槽，Semformer 学习特殊 token 作为规划信号。
    *   **Latent Reasoning (LR)**：将自回归过程转移到连续潜在空间，生成潜在向量序列后再解码为文本，解决 token 粒度问题；如 SentenceVAE 按句生成潜在向量，Coconut 在文本段间进行潜在推理。
    *   **Continuous Generation Approaches (CG)**：放弃逐 token 生成，通过全局并行优化（如扩散模型、流匹配、能量模型）迭代精炼整个输出，支持全局规划和双向修正；如 Diffusion-LM 在嵌入空间去噪，Mercury 实现高速并行去噪。
    *   **Non-Transformer Architectures (NTA)**：采用非 Transformer 架构（如状态空间模型、联合嵌入预测架构），通过不同序列建模机制从根本上绕过 NTP；如 Mamba 实现线性时间推理，LLM-JEPA 在共享语义空间中预测。
*   **分析重点**：每类方法均从理论基础、实现方式、优缺点及适用场景进行详细讨论，旨在为研究者提供清晰的分类和研究方向。

## Experiment

*   **有效性**：由于是综述性论文，作者未进行独立实验，而是引用原始研究结果；如 MTP 方法（如 Gloeckle et al.）报告推理速度提升 3 倍，PtG 在长文本任务中改善结构化输出，CG 方法（如 Mercury）实现 1100 tokens/sec 的生成速度，显示出并行生成的潜力。
*   **局限性**：许多方法仍存在不足，如 MTP 缺乏全局规划深度，LR 和 CG 方法在可解释性和训练复杂性上挑战较大，NTA 的应用范围和泛化能力有限，整体上与自回归模型在流畅性上仍有差距。
*   **实验设置合理性**：论文覆盖了 2020-2025 年的主要研究，分类逻辑清晰，引用广泛，但未进行统一基准测试，依赖原始论文结果，可能存在比较标准不一致的问题。

## Further Thoughts

论文启发我们重新思考语言生成的粒度，未来可探索多层次语义单位（如词、句、段）的混合建模；PtG 的两阶段设计提示引入显式规划模块或人类输入计划以提升可控性；CG 和 NTA 的非自回归范式表明摆脱 NTP 限制可能带来计算效率和长程依赖建模的突破，值得进一步探索非 Transformer 架构在开放生成任务中的潜力。