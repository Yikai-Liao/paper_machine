---
title: "Unifying Multimodal Large Language Model Capabilities and Modalities via Model Merging"
pubDatetime: 2025-05-26T12:23:14+00:00
slug: "2025-05-mllm-merging-benchmark"
type: "arxiv"
id: "2505.19892"
score: 0.8341543865006648
author: "grok-3-latest"
authors: ["Yongxian Wei", "Runxi Cheng", "Weike Jin", "Enneng Yang", "Li Shen", "Lu Hou", "Sinan Du", "Chun Yuan", "Xiaochun Cao", "Dacheng Tao"]
tags: ["LLM", "Multimodal Learning", "Model Merging", "Task Vector", "Optimization"]
institution: ["Tsinghua University", "Huawei Noah’s Ark Lab", "Sun Yat-sen University", "Nanyang Technological University"]
description: "本文提出一个针对多模态大语言模型的模型融合基准，并通过新型任务向量优化方法显著提升融合性能，为构建高效、强大的统一模型提供了可扩展的解决方案。"
---

> **Summary:** 本文提出一个针对多模态大语言模型的模型融合基准，并通过新型任务向量优化方法显著提升融合性能，为构建高效、强大的统一模型提供了可扩展的解决方案。 

> **Keywords:** LLM, Multimodal Learning, Model Merging, Task Vector, Optimization

**Authors:** Yongxian Wei, Runxi Cheng, Weike Jin, Enneng Yang, Li Shen, Lu Hou, Sinan Du, Chun Yuan, Xiaochun Cao, Dacheng Tao

**Institution(s):** Tsinghua University, Huawei Noah’s Ark Lab, Sun Yat-sen University, Nanyang Technological University


## Problem Background

基础模型由于资源密集的训练需求更新缓慢，而领域特定模型在更新间隙不断改进；模型融合（Model Merging）旨在将多个专家模型合并为一个统一模型，以降低存储和部署成本并支持去中心化开发。然而，针对多模态大语言模型（MLLM）的融合研究缺乏系统性基准，特别是在整合多种任务能力（如视觉问答、几何推理）和不同模态（如视觉-语言、音频-语言）方面存在挑战。本文试图解决如何在不依赖额外数据训练的情况下，通过模型融合技术构建更强大的 MLLM。

## Method

* **基准构建**：为 MLLM 设计了一个包含多种任务（VQA、Geometry、Chart、OCR、Grounding）的模型融合基准，提供了 LoRA 和全参数微调的检查点，并探索了跨模态融合（Omni-language Model），以支持能力整合和模态整合。
* **现有方法对比**：实现了 10 种模型融合算法，分为线性插值（如 Weight Averaging）、稀疏化（如 TIES-Merging）、SVD 基方法（如 TSV Merging）和优化基方法（如 WUDI Merging），为性能评估提供了全面基础。
* **新型方法（WUDI v2）**：提出了一种改进的任务向量优化方法，通过低秩近似（Low-Rank Approximation）去除任务向量中的冗余噪声，并基于任务向量交互定义损失函数进行优化。具体而言：
  * 对于全参数微调，通过 SVD 分解提取任务向量的核心知识，减少任务间干扰。
  * 对于 LoRA 微调，采用 SGD 优化器替代 Adam，通过初始化合并向量为任务向量均值，并施加低秩约束，稳定优化过程，避免合并向量范数过大导致模型性能崩溃。
* **核心创新**：新方法在不修改原始模型架构的前提下，通过数学优化手段提升融合模型性能，同时保持计算效率，适用于资源受限场景。

## Experiment

* **能力融合效果**：在 InternVL2.5 和 Qwen2-VL 模型上，融合后的模型在多个任务上显著优于个体专家模型，例如 Qwen2-VL 融合模型在 Geometry 任务上的表现（51.05 和 40.79）远超个体模型（42.50 和 28.95）；新方法 WUDI v2 平均性能提升 2.48%，在多个任务上达到最佳或次佳表现。
* **模态融合效果**：融合视觉、音频和视频模态的模型在零样本任务（如 MUSIC-AVQA 和 AVQA）上优于单一模态模型，WUDI v2 甚至超越在线组合方法，显示出模态互补性带来的显著提升。
* **实验设置合理性**：实验覆盖全参数微调和 LoRA 微调两种场景，数据集规模大（每个任务至少 100k 样本），评价指标全面（涵盖多种任务和模态），并与混合训练进行了对比；此外，测试了 Hugging Face 上的实际模型，验证了方法的实用性。
* **计算效率**：融合方法计算成本远低于传统训练，例如 InternVL2.5 融合仅需 0.22 小时和 2.62GB GPU 内存，而混合训练需 25.38 小时和 240GB 内存。
* **局限性**：由于资源限制，实验仅限于 7B 参数模型，数据集质量可能不均，未来可扩展到更大规模模型和多语言任务。

## Further Thoughts

论文展现了数据无关的模型融合潜力，启发我们思考任务向量是否能在不同模型架构间迁移，以实现跨架构能力整合；多模态互补性提示是否可融合更多模态（如触觉、传感器数据）以提升泛化能力；低秩近似和初始化策略的有效性启发是否可通过其他数学工具（如张量分解）进一步优化任务向量；此外，模型融合支持去中心化开发，是否可构建一个动态更新的‘模型生态系统’，让开源社区持续贡献和整合模型？