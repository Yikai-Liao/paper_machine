---
title: "Unifying Multimodal Large Language Model Capabilities and Modalities via Model Merging"
pubDatetime: 2025-05-26T12:23:14+00:00
slug: "2025-05-mllm-merging-benchmark"
type: "arxiv"
id: "2505.19892"
score: 0.8341543865006648
author: "grok-3-latest"
authors: ["Yongxian Wei", "Runxi Cheng", "Weike Jin", "Enneng Yang", "Li Shen", "Lu Hou", "Sinan Du", "Chun Yuan", "Xiaochun Cao", "Dacheng Tao"]
tags: ["Multimodal LLM", "Model Merging", "Task Vector", "Capability Integration", "Modality Fusion"]
institution: ["Tsinghua University", "Huawei Noah’s Ark Lab", "Sun Yat-sen University", "Nanyang Technological University"]
description: "本文通过构建多模态大语言模型（MLLMs）融合基准并提出优化任务向量的新方法，展示了模型融合在提升多任务能力和多模态能力方面的潜力，同时以较低计算成本实现与混合训练相当或更优的性能。"
---

> **Summary:** 本文通过构建多模态大语言模型（MLLMs）融合基准并提出优化任务向量的新方法，展示了模型融合在提升多任务能力和多模态能力方面的潜力，同时以较低计算成本实现与混合训练相当或更优的性能。 

> **Keywords:** Multimodal LLM, Model Merging, Task Vector, Capability Integration, Modality Fusion

**Authors:** Yongxian Wei, Runxi Cheng, Weike Jin, Enneng Yang, Li Shen, Lu Hou, Sinan Du, Chun Yuan, Xiaochun Cao, Dacheng Tao

**Institution(s):** Tsinghua University, Huawei Noah’s Ark Lab, Sun Yat-sen University, Nanyang Technological University


## Problem Background

基础模型由于资源密集的训练需求更新缓慢，而领域特定模型在更新间隙不断改进，模型融合（Model Merging）旨在将多个专家模型合并为一个统一模型，以降低存储和部署成本并支持去中心化开发。
然而，当前多模态大语言模型（MLLMs）缺乏清晰划分任务的融合基准，且如何高效融合不同模态（如视觉-语言、音频-语言、视频-语言）以构建全能语言模型（Omni-language Model）仍是一个挑战。

## Method

*   **基准构建**：为多模态大语言模型（MLLMs）设计了一个模型融合基准，涵盖多种任务（如视觉问答 VQA、几何、图表、OCR 和定位 Grounding），为每种任务收集至少 10 万样本的公开数据集，并选择 InternVL2.5 和 Qwen2-VL 两种视觉-语言模型，分别进行全微调（Full Fine-Tuning）和低秩适配（LoRA Fine-Tuning），提供相应的检查点以评估融合方法的泛化性。
*   **任务向量优化**：提出了一种新的融合方法，通过优化任务向量（Task Vector，即微调模型与基础模型之间的参数变化）来提升融合效果。具体包括：
    *   对于全微调模型，通过低秩近似（Low-Rank Approximation）去除任务向量中的冗余噪声，利用奇异值分解（SVD）提取核心任务知识，优化合并向量以减少任务间干扰。
    *   对于 LoRA 微调模型，针对其低秩特性带来的优化挑战，采用随机梯度下降（SGD）优化器替代 Adam 以提高稳定性，并通过初始化合并向量为任务向量均值及直接低秩近似控制合并向量范数，防止模型能力崩溃。
*   **模态融合策略**：探索将不同模态（如视觉、音频、视频）的模型融合到共享的大语言模型中，保留各自模态编码器和连接器，仅合并语言模型参数，实现数据无关（Data-Free）的静态融合，避免动态融合或测试时适应的额外存储和计算开销。

## Experiment

*   **能力融合效果**：在 InternVL2.5 和 Qwen2-VL 模型上，融合后的模型在多个任务（如几何、图表、OCR）上的表现显著优于单个专家模型，例如 Qwen2-VL 在几何任务上的表现从 42.50 和 28.95 提升到 51.05 和 40.79；作者提出的方法（WUDI v2）平均性能提升了 2.48%，在多个任务上达到最佳或次佳表现。
*   **模态融合效果**：融合视觉、音频和视频模态的模型在零样本图像-音频-视频问答任务（如 MUSIC-AVQA 和 AVQA）上平均性能优于单一模态模型，显示出模态信息的互补性。
*   **实验设置合理性**：实验对比了 10 种融合方法，并与混合训练（Mixture Training）进行了比较，证明了模型融合在计算成本和时间上的优势（例如 InternVL2.5 融合仅需 0.22 小时和 2.62GB GPU 内存，而混合训练需 25.38 小时和 240GB 内存）；此外，测试了 Hugging Face 上的实际微调模型，验证了方法的实用性。
*   **局限性**：由于资源限制，实验仅限于 7B 参数模型，可能影响对更大规模模型的泛化性评估。

## Further Thoughts

模型融合作为一种无需额外数据训练的解决方案，为资源受限团队提供了降低 MLLMs 开发成本的新思路，是否可以通过‘模块化’设计将不同能力或模态作为可插拔组件动态组合？任务向量优化揭示了微调参数变化对融合效果的影响，是否可以设计自适应融合策略，根据任务或模态特性动态调整融合权重？模态融合展示了多模态信息的互补性，未来是否可以通过融合更多模态（如触觉、传感器数据）进一步提升模型全能性，特别是在机器人或物联网领域的应用？