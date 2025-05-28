---
title: "Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models"
pubDatetime: 2025-05-26T08:53:02+00:00
slug: "2025-05-residual-alignment-sampling"
type: "arxiv"
id: "2505.19700"
score: 0.8669710442341461
author: "grok-3-latest"
authors: ["Yi Liu", "Dianqing Liu", "Mingye Zhu", "Junbo Guo", "Yongdong Zhang", "Zhendong Mao"]
tags: ["LLM", "Alignment", "Sampling", "Residual Correction", "Efficiency"]
institution: ["State Key Laboratory of Communication Content Cognition, People’s Daily Online, Beijing, China", "University of Science and Technology of China, Hefei, China"]
description: "本文提出 Residual Alignment Model (RAM)，通过重要性采样分离对齐模块与预训练模型，实现高效灵活的对齐，并利用令牌级解码策略减少推理延迟。"
---

> **Summary:** 本文提出 Residual Alignment Model (RAM)，通过重要性采样分离对齐模块与预训练模型，实现高效灵活的对齐，并利用令牌级解码策略减少推理延迟。 

> **Keywords:** LLM, Alignment, Sampling, Residual Correction, Efficiency

**Authors:** Yi Liu, Dianqing Liu, Mingye Zhu, Junbo Guo, Yongdong Zhang, Zhendong Mao

**Institution(s):** State Key Laboratory of Communication Content Cognition, People’s Daily Online, Beijing, China, University of Science and Technology of China, Hefei, China


## Problem Background

大型语言模型（LLM）在各行业的广泛应用增加了对高质量、可定制化输出的需求，但传统对齐方法需要对整个大模型进行资源密集的重新训练，缺乏灵活性，难以快速适应不同领域需求和人类价值观，同时推理过程中常伴随首词延迟等问题。

## Method

*   **核心框架：Residual Alignment Model (RAM)**：将对齐过程形式化为重要性采样，其中未对齐的预训练模型（Proposal Module）作为提议分布，对齐过程通过一个自回归的 Residual Aligner 模块进行二次采样，估计重要性权重，最终通过线性组合生成对齐模型 P_θ(y|x) = P_M(y|x) * Q_θ(y|x) / Z_θ(x)。
*   **模块分离与灵活性**：通过将 Residual Aligner 从目标对齐模型中分离，仅需训练较小的对齐模块，而保持大模型冻结，从而实现资源高效的对齐，并支持多个对齐模块共享同一 Proposal Module，提升跨领域资源利用率。
*   **序列级训练策略**：提出一种高效训练方法，仅优化 Residual Aligner，利用监督微调（SFT）目标函数，通过 Jensen 不等式推导下界，并引入拉格朗日乘子法和参数 α 平衡对齐模块与提议模块的影响，训练过程中 Proposal Module 仅用于一次性数据合成或保持未使用。
*   **令牌级解码优化**：针对首词延迟问题，设计 Proposing-Aligning-Reducing Sampling 策略，利用自回归特性避免直接估计分区函数，通过核采样提出候选令牌、基于重要性权重对齐、并归一化后采样最终令牌，同时结合 KL 散度判断分布差异以避免性能下降。

## Experiment

*   **性能提升显著**：在 LLaMA 3 和 Qwen 2.5 模型家族上，RAM 在指令跟随（UltraChat）、领域适应（TL;DR Summarization）和偏好优化（Anthropic-HH）任务中表现出色，胜率（win rate）分别平均提升 20%、7% 和 5%-9.2%，尤其在资源受限环境下优于基线方法（如 SFT、DPO 和 Aligner）。
*   **稳定性与对比优势**：相比 Aligner，RAM 避免了参考响应的分布外（OOD）问题，性能更稳定；相比传统方法，训练效率提升显著（如 DPO 任务中效率提升 13.33 倍）。
*   **实验设置全面**：实验涵盖多种任务、数据集和模型规模（0.5B-70B），使用 Qwen2.5 和 GPT-4 作为评判模型，评估指标包括长度控制胜率（LC）和原始胜率（WR），消融研究验证了 Residual Aligner 规模和参数 α 的影响，整体设计合理且结果可信。
*   **局限性**：词汇表一致性要求限制了方法适用性，Residual Aligner 规模增加带来的性能提升幅度较小（平均 2.1%-2.4%），未来潜力需进一步探索。

## Further Thoughts

模块化对齐的思路启发我们思考是否可以构建一个通用的对齐模块库，供不同大模型共享使用；重要性采样的应用是否能结合其他统计采样方法（如蒙特卡洛方法）以应对分布差异较大的场景；令牌级解码策略是否可推广至其他生成任务以提升推理效率。