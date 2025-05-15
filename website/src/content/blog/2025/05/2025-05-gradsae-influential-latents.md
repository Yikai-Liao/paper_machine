---
title: "Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders"
pubDatetime: 2025-05-12T21:29:12+00:00
slug: "2025-05-gradsae-influential-latents"
type: "arxiv"
id: "2505.08080"
score: 0.46098898047306625
author: "grok-3-latest"
authors: ["Dong Shu", "Xuansheng Wu", "Haiyan Zhao", "Mengnan Du", "Ninghao Liu"]
tags: ["LLM", "Sparse Autoencoder", "Model Steering", "Gradient Analysis", "Interpretability"]
institution: ["Northwestern University", "University of Georgia", "New Jersey Institute of Technology"]
description: "本文提出 Gradient Sparse Autoencoder (GradSAE)，通过输出侧梯度信息准确识别对大型语言模型输出有关键影响的潜在特征，提升了模型解释与操控的精度。"
---

> **Summary:** 本文提出 Gradient Sparse Autoencoder (GradSAE)，通过输出侧梯度信息准确识别对大型语言模型输出有关键影响的潜在特征，提升了模型解释与操控的精度。 

> **Keywords:** LLM, Sparse Autoencoder, Model Steering, Gradient Analysis, Interpretability

**Authors:** Dong Shu, Xuansheng Wu, Haiyan Zhao, Mengnan Du, Ninghao Liu

**Institution(s):** Northwestern University, University of Georgia, New Jersey Institute of Technology


## Problem Background

大型语言模型（LLMs）的内部表示解释与操控是一个核心挑战。稀疏自编码器（SAEs）作为一种解释工具，通过学习过完备的稀疏潜在空间来分解模型特征，但传统方法仅依赖输入侧激活（input activations）来识别潜在特征，假设这些激活对输出有直接因果影响，而这一假设未经证实，可能导致模型操控（steering）时出现意外效果。本文旨在解决如何准确识别对模型输出具有真正因果影响的潜在特征，并利用这些特征实现更可靠的模型操控。

## Method

* **核心思想**：提出 Gradient Sparse Autoencoder (GradSAE)，通过结合输出侧梯度信息，评估潜在特征（latents）对模型输出的因果影响，而非仅依赖输入侧激活。
* **具体实现**：
  * 在 SAE 的潜在激活（H）基础上，计算输出预测概率（logits）对每个潜在特征的梯度，通过泰勒展开近似估计其影响（定义为影响分数 g_n,c）。
  * 结合梯度信号和原始激活值，计算每个潜在特征的综合影响分数，并按此分数排序，筛选出高影响（TopK）和低影响（BottomK）潜在特征。
  * 在推理时，通过屏蔽或替换潜在特征，验证其对输出的影响，或进行模型操控，无需重新训练 SAE 或 LLM，是一种高效的训练无关（training-free）方法。
* **关键创新**：相比传统仅基于输入激活的方法，GradSAE 引入输出侧梯度信号，提供更精确的影响力评估，适用于任何指令微调的 LLM 和 SAE。

## Experiment

* **有效性**：在扰动实验中，屏蔽 GradSAE 识别的高影响（TopK）潜在特征导致模型性能显著下降（例如在 SQuAD 数据集上，K=50% 时 Exact Match 从 100% 降至 30.58%），而屏蔽低影响（BottomK）特征几乎无影响（EM 接近 100%），验证了潜在特征影响的不均等性。
* **优越性**：相比基线方法（仅基于输入激活），GradSAE 在屏蔽 TopK 特征时性能下降更明显（例如 K=50% 时基线 EM 为 53.42%，GradSAE 为 30.58%），表明其更准确地识别了关键特征；在局部操控实验中，GradSAE 的 TopK 特征替换后操控效果更强（K=10 时 F1 达 10.21%，基线为 6.69%）。
* **实验设置合理性**：实验基于 SQuAD 数据集，覆盖 Gemma 2 9B 和 LLaMA 3 8B 模型的不同层（Layer 9, 20, 31），结果一致，表明方法具有泛化性；评价指标（Exact Match 和 F1）全面，数据选择适合测试操控效果。
* **开销**：方法为训练无关，仅在推理时计算梯度，计算开销较小，适合实际应用。

## Further Thoughts

GradSAE 的梯度引导方法启发我们思考，是否可以进一步结合上下文依赖的梯度信号，动态调整潜在特征的选择，以适应不同任务需求？此外，这种训练无关的特征评估思路是否能扩展到其他领域（如视觉模型），通过输出反馈识别关键表示？另一个方向是，是否可以通过多任务输出的梯度分析，识别对特定任务有针对性影响的潜在特征，从而实现更精细的任务操控？