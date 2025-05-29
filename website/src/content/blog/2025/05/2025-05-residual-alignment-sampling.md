---
title: "Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models"
pubDatetime: 2025-05-26T08:53:02+00:00
slug: "2025-05-residual-alignment-sampling"
type: "arxiv"
id: "2505.19700"
score: 0.8669710442341461
author: "grok-3-latest"
authors: ["Yi Liu", "Dianqing Liu", "Mingye Zhu", "Junbo Guo", "Yongdong Zhang", "Zhendong Mao"]
tags: ["LLM", "Alignment Module", "Importance Sampling", "Residual Correction", "Token Decoding"]
institution: ["State Key Laboratory of Communication Content Cognition, People’s Daily Online, Beijing, China", "University of Science and Technology of China, Hefei, China"]
description: "本文提出基于重要性采样的 *Residual Alignment Model*，通过模块化分离和高效训练策略，实现大型语言模型的高效对齐，并通过令牌级解码显著降低推理延迟。"
---

> **Summary:** 本文提出基于重要性采样的 *Residual Alignment Model*，通过模块化分离和高效训练策略，实现大型语言模型的高效对齐，并通过令牌级解码显著降低推理延迟。 

> **Keywords:** LLM, Alignment Module, Importance Sampling, Residual Correction, Token Decoding

**Authors:** Yi Liu, Dianqing Liu, Mingye Zhu, Junbo Guo, Yongdong Zhang, Zhendong Mao

**Institution(s):** State Key Laboratory of Communication Content Cognition, People’s Daily Online, Beijing, China, University of Science and Technology of China, Hefei, China


## Problem Background

大型语言模型（LLMs）在各行业的广泛应用增加了对高质量、可定制输出的需求，但传统对齐方法需要对大型预训练模型进行整体重新训练，资源消耗高且灵活性差；此外，现有方法在推理时存在首词延迟问题和分布外输入风险，亟需一种高效、灵活的对齐解决方案。

## Method

* **核心思想**：提出 *Residual Alignment Model* (RAM)，将对齐过程形式化为重要性采样，通过模块化设计分离对齐模块与预训练模型，实现高效对齐。
* **具体实现**：
  * 将未对齐的预训练模型命名为 *Proposal Module*，作为提议分布，负责生成初始候选输出。
  * 引入一个自回归的 *Residual Aligner* 模块，作为重要性权重的估计器，对提议分布进行二次采样，生成对齐后的输出。
  * 通过线性组合两模块的概率分布构建最终对齐模型 *P_θ(y|x)*，实现对齐模块的自然分离。
  * **训练策略**：仅对较小的 *Residual Aligner* 进行序列级训练，冻结大型 *Proposal Module*，通过监督微调（SFT）目标优化损失函数，并引入参数 *α* 调节提议分布的影响，显著降低计算成本。
  * **推理优化**：设计 *Proposing-Aligning-Reducing Sampling* 策略，利用自归一化重要性采样进行令牌级解码，减少首词延迟；具体步骤包括从 *Proposal Module* 提出候选令牌、由 *Residual Aligner* 分配重要性权重、最终归一化采样选择目标令牌。
* **关键点**：不需整体微调大模型，仅通过小规模模块训练实现对齐，同时优化推理效率，避免分布外输入问题。

## Experiment

* **有效性**：在 LLaMA 3 和 Qwen 2.5 模型家族上，RAM 在指令跟随（UltraChat）、领域适应（TL;DR Summarization）和偏好优化（Anthropic-HH）任务中表现出显著提升，例如 UltraChat 上平均胜率提升 20%，Anthropic-HH 上 DPO 模型胜率提升高达 9.2%。
* **优越性**：相比基线方法（如 Aligner 和传统 SFT），RAM 在低参数设置下避免了过拟合和重复模式问题，直接建模条件概率 P(y|x) 降低了分布外输入风险，性能更稳定。
* **实验设置**：实验覆盖多种任务和数据集，使用 AlpacaEval 2 框架以 Qwen2.5 和 GPT-4 作为评判模型，确保客观性；消融研究验证了 *Residual Aligner* 大小和参数 *α* 的影响，方法鲁棒性强。
* **效率**：训练效率显著优于传统方法，例如在 DPO 任务中效率提升达 13.33 倍，适合资源受限环境。

## Further Thoughts

模块化对齐思想启发我们是否可以构建一个通用的对齐模块库，供不同大模型共享，降低重复训练成本；重要性采样在 NLP 中的应用是否可扩展至文本风格迁移或生成多样性控制；令牌级解码策略是否能结合自适应采样技术进一步提升生成质量和效率。