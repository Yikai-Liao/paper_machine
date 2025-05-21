---
title: "Reward Inside the Model: A Lightweight Hidden-State Reward Model for LLM's Best-of-N sampling"
pubDatetime: 2025-05-18T04:00:35+00:00
slug: "2025-05-hidden-state-reward"
type: "arxiv"
id: "2505.12225"
score: 0.7932893227673858
author: "grok-3-latest"
authors: ["Jizhou Guo", "Zhaomin Wu", "Philip S. Yu"]
tags: ["LLM", "Hidden States", "Reward Model", "Sampling", "Reasoning"]
institution: ["Shanghai Jiao Tong University", "National University of Singapore", "University of Illinois at Chicago"]
description: "本文提出了一种基于隐藏状态的轻量级奖励模型 ELHSR，以极低的参数量和计算成本显著提升了大型语言模型在 Best-of-N 采样中的推理性能。"
---

> **Summary:** 本文提出了一种基于隐藏状态的轻量级奖励模型 ELHSR，以极低的参数量和计算成本显著提升了大型语言模型在 Best-of-N 采样中的推理性能。 

> **Keywords:** LLM, Hidden States, Reward Model, Sampling, Reasoning

**Authors:** Jizhou Guo, Zhaomin Wu, Philip S. Yu

**Institution(s):** Shanghai Jiao Tong University, National University of Singapore, University of Illinois at Chicago


## Problem Background

大型语言模型（LLMs）在复杂推理任务中的性能提升依赖于高质量的奖励模型，尤其是在 Best-of-N 采样策略中。然而，传统奖励模型通常基于文本输出，参数量庞大、计算成本高，且需要大量训练数据，限制了其在资源受限环境下的应用。作者旨在利用 LLM 内部隐藏状态（Hidden States）中蕴含的丰富信息，开发一种轻量级、高效的奖励模型，以解决传统方法在效率和数据需求上的瓶颈。

## Method

* **核心思想**：提出 Efficient Linear Hidden State Reward (ELHSR) 模型，利用 LLM 的隐藏状态构建一个参数高效的奖励模型，直接从内部表示中提取推理路径的正确性信号，而非依赖文本输出。
* **具体实现**：
  - **隐藏状态提取**：从 LLM 的每一层中提取每个 token 的隐藏状态（Hidden States），并将所有层的隐藏状态拼接为一个高维向量，表示每个 token 的内部信息。
  - **线性变换**：对每个 token 的隐藏状态向量应用线性变换，生成两个输出：一个是门控值（Gating Value）的预激活值，另一个是 token 级别的奖励值（Token-Level Reward）。
  - **门控机制**：通过 sigmoid 函数处理门控预激活值，得到门控值，用于动态加权每个 token 对最终奖励的贡献，突出重要 token 的作用。
  - **奖励计算**：基于门控值对所有 token 的奖励值进行加权平均，得到整个推理路径的最终奖励分数，用于 Best-of-N 采样中选择最佳路径。
  - **训练方式**：采用二元交叉熵损失（Binary Cross-Entropy Loss）进行训练，标签为推理路径是否正确，优化目标是最大化正确路径的选择概率。
* **扩展特性**：
  - 支持仅基于 logits 训练，适用于无法访问隐藏状态的闭源模型。
  - 可与传统奖励模型结合，通过排名选择或加权平均策略进一步提升性能。
* **效率优势**：ELHSR 参数量极低（仅为传统模型的 0.005%），训练数据需求少，计算开销小（时间和 FLOPs 降低几个数量级）。

## Experiment

* **有效性**：ELHSR 在 MATH、GSM8K、AQuA_RAT 和 CRUXEval-O 等数据集上，结合 Llama-3.2-3B、Llama-3.1-8B 和 Ministral-8B 等模型进行测试，结果显示其在 Best-of-N 采样中的准确率显著优于基线奖励模型（如 EurusRM-7B、UltraRM-13B）。例如，在 MATH 数据集上，ELHSR 平均准确率为 57.5%，高于最佳基线的 52.9%。
* **效率**：ELHSR 参数量仅为基线的 0.005%，训练数据需求低至 6000 样本，计算效率（时间和 FLOPs）提升了几个数量级，适合资源受限环境。
* **扩展性**：实验表明 ELHSR 具有良好的训练时和测试时扩展性，性能随训练样本和推理路径数量增加而持续提升；即使仅使用部分层隐藏状态或 logits 训练，性能仍接近或优于全层模型。
* **合理性**：实验设置全面，涵盖多个数据集和模型，基线选择合理（包括领先的开源模型和微调模型），评估方法考虑了数学表达式的语义等价性，数据划分和早停策略也有效避免了过拟合。

## Further Thoughts

ELHSR 利用隐藏状态的线性表示构建奖励模型，这一思路启发我们可以在其他任务（如安全对齐、幻觉检测、知识编辑）中探索内部表示的应用，以提升 LLM 的效率和可解释性；此外，门控机制动态加权 token 重要性的设计，可能对识别复杂推理任务中的关键步骤有深远影响；最后，内部信号与外部信号结合的策略表明，未来可以进一步研究多源信号的互补性，以在不同资源条件下优化模型性能。