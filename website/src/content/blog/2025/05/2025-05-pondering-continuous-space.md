---
title: "Pretraining Language Models to Ponder in Continuous Space"
pubDatetime: 2025-05-27T03:47:33+00:00
slug: "2025-05-pondering-continuous-space"
type: "arxiv"
id: "2505.20674"
score: 0.9051415742015179
author: "grok-3-latest"
authors: ["Boyi Zeng", "Shixiang Song", "Siyuan Huang", "Yixuan Wang", "He Li", "Ziwei He", "Xinbing Wang", "Zhiyu Li", "Zhouhan Lin"]
tags: ["LLM", "Test Time Scaling", "Pre-Training", "Reasoning", "Continuous Space"]
institution: ["Shanghai Jiao Tong University", "Institute for Advanced Algorithms Research", "Shanghai Innovation Institute"]
description: "本文通过自监督学习引入 pondering 机制，在预训练阶段通过连续嵌入空间的迭代精炼显著提升语言模型性能，实现了更高的参数和数据效率。"
---

> **Summary:** 本文通过自监督学习引入 pondering 机制，在预训练阶段通过连续嵌入空间的迭代精炼显著提升语言模型性能，实现了更高的参数和数据效率。 

> **Keywords:** LLM, Test Time Scaling, Pre-Training, Reasoning, Continuous Space

**Authors:** Boyi Zeng, Shixiang Song, Siyuan Huang, Yixuan Wang, He Li, Ziwei He, Xinbing Wang, Zhiyu Li, Zhouhan Lin

**Institution(s):** Shanghai Jiao Tong University, Institute for Advanced Algorithms Research, Shanghai Innovation Institute


## Problem Background

当前大型语言模型（LLMs）性能提升主要依赖参数和数据规模的扩展，但面临数据枯竭、计算成本高昂及性能饱和等瓶颈。
作者受人类在复杂问题上通过反复思考（pondering）提升能力的启发，提出在预训练阶段引入类似思考过程，以在不增加参数规模的情况下提升模型性能，同时解决现有测试时扩展方法（如链式思维 CoT）对小模型效果不佳及对离散词汇表依赖的局限性。

## Method

*   **核心思想:** 在单个 token 生成步骤中，通过多次迭代前向计算模拟‘思考’过程（pondering），精炼模型预测分布，提升性能。
*   **具体实现:** 
    *   在每次 token 预测时，不直接从概率分布中采样离散 token，而是根据预测概率对所有 token 嵌入进行加权求和，生成一个连续的‘pondering embedding’。
    *   将该连续嵌入与原始输入嵌入通过残差连接相加，作为新的输入再次送入语言模型进行前向计算。
    *   重复上述过程 k 次（即 pondering steps），逐步精炼预测分布，最终基于最后一步的概率计算交叉熵损失并优化模型。
*   **优化细节:** 为降低计算复杂度，仅使用 top-K 高概率 token 计算 pondering embedding，确保计算开销可控。
*   **优势:** 该方法完全基于自监督学习，无需额外标注数据或强化学习，可无缝集成到现有语言模型架构中，并通过连续嵌入突破离散词汇表的表达限制。

## Experiment

*   **有效性:** Pondering 模型在语言建模任务上的困惑度（perplexity）显著优于同规模 vanilla 模型，例如 PonderingPythia-1B 性能接近官方 Pythia-1B 的两倍参数规模，或仅用 41% 训练 token 达到相似效果。
*   **下游任务表现:** 在 9 个下游任务（如 LAMBADA, PIQA）上，PonderingPythia 在零样本和五样本设置下均显著优于官方 Pythia 模型，PonderingPythia-1B 甚至接近训练数据量多 10 倍的 TinyLlama-1.1B。
*   **可扩展性:** 增加 pondering steps（从 0 到 10）持续降低语言建模损失，表明方法潜力巨大。
*   **实验设置合理性:** 实验覆盖多种架构（GPT-2, LLaMA, Pythia）、不同规模（14M 到 1.4B 参数）及大规模预训练（Pile 数据集 300B token），对比多种基线（OPT, Bloom），较为全面，但受限于计算资源未测试更大规模模型。
*   **开销分析:** 推理开销随 pondering steps 线性增加，可能限制实际部署应用。

## Further Thoughts

通过连续嵌入空间进行‘思考’的机制启发我们探索其他领域（如图像生成或多模态模型）中类似中间状态迭代精炼的可能性；此外，pondering 与传统参数扩展和推理时扩展（如 CoT）正交的特性提示可以尝试多维度扩展策略组合；最后，pondering embedding 的语义解释为未来研究模型内部‘思考’过程提供了有趣方向。