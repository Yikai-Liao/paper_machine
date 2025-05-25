---
title: "Latent Principle Discovery for Language Model Self-Improvement"
pubDatetime: 2025-05-22T17:20:18+00:00
slug: "2025-05-self-taught-principles"
type: "arxiv"
id: "2505.16927"
score: 0.8031390775048035
author: "grok-3-latest"
authors: ["Keshav Ramji", "Tahira Naseem", "Ramón Fernandez Astudillo"]
tags: ["LLM", "Self-Correction", "Latent Variable", "Clustering", "Reasoning"]
institution: ["IBM Research AI"]
description: "本文提出 STaPLe 方法，通过自动化发现和学习潜在原则，显著提升小型语言模型在指令跟随任务上的自我改进能力，并通过聚类提高可解释性。"
---

> **Summary:** 本文提出 STaPLe 方法，通过自动化发现和学习潜在原则，显著提升小型语言模型在指令跟随任务上的自我改进能力，并通过聚类提高可解释性。 

> **Keywords:** LLM, Self-Correction, Latent Variable, Clustering, Reasoning

**Authors:** Keshav Ramji, Tahira Naseem, Ramón Fernandez Astudillo

**Institution(s):** IBM Research AI


## Problem Background

现代语言模型（LMs）在生成高质量响应时，难以满足多重且可能重叠的人类定义标准，传统方法依赖人工标注或静态‘宪法’（Constitutional AI）指导模型行为，成本高且适应性差。
论文旨在自动化发现语言模型自我改进所需的潜在原则（Latent Principles），以减少对人类干预的依赖，并在开放性文本生成任务中实现自我校正（Self-Correction）。

## Method

*   **核心思想:** 提出‘Self-Taught Principle Learning (STaPLe)’方法，通过模型自身挖掘潜在原则并学习如何调用这些原则改进响应，实现自我改进。
*   **具体实现:** 
    *   **原则发现阶段（E-step）**：基于蒙特卡洛期望最大化（MC-EM）算法，利用模型生成初始响应后，通过‘提示’金标准响应（Gold Response）采样多个潜在原则（Principles），并生成改进后的响应；采用拒绝采样（Rejection Sampling）选择最接近金标准的原则-响应对。
    *   **原则学习阶段（M-step）**：对采样的原则-响应轨迹进行监督微调（Supervised Fine-Tuning），训练模型在推理时根据输入和初始响应调用合适原则并改进输出。
    *   **后验正则化与聚类**：通过层次聚类（Hierarchical Clustering）压缩原则集合，形成人类可读的‘宪法’（Constitution），减少冗余并提高可解释性，同时避免性能下降。
    *   **迭代改进**：重复上述步骤，逐步提升模型性能，直到性能饱和。
*   **关键创新:** 将原则视为潜在变量（Latent Variable），通过模型自生成和自学习减少外部监督，同时结合聚类技术平衡性能与可解释性。

## Experiment

*   **有效性:** 在 7-8B 参数规模的小型语言模型（如 Llama3.1-8B-Instruct、Granite-3.1-8B-Instruct、Qwen2.5-7B-Instruct）上，STaPLe 迭代后在 AlpacaEval 胜率提升 +8-10%，MT-Bench 平均得分提升 +0.3，IFEval 原则跟随胜率提升 +19-23%，显著优于基线方法（如 Self-Refine 和 STaR）。
*   **优越性:** 相比基线，STaPLe 在多轮迭代后性能提升更明显，尤其在 MT-Bench Turn 2 得分上（平均 +0.22），显示出更强的多轮对话自我校正能力；聚类版本（Constrained STaPLe）在部分指标上甚至优于未聚类版本。
*   **实验设置合理性:** 实验覆盖多个模型和混合领域数据集（如 Anthropic HH-RLHF、UltraFeedback），通过多轮迭代验证自我改进可持续性；评估指标细致（如 Prometheus-v2.0 评判），结果可信。
*   **局限性:** 性能在第 3-4 轮迭代后趋于饱和，部分模型（如 Llama-8B 和 Granite-8B）在第 4 轮略有下降，表明自我改进存在上限。

## Further Thoughts

STaPLe 的自动化原则发现机制启发我们可以在其他领域（如图像生成或多模态任务）中探索潜在规则的挖掘；此外，是否可以通过引入外部知识库或多模型协作来丰富原则多样性，例如利用专门的‘原则生成模型’辅助目标模型，可能进一步提升原则质量和覆盖范围。