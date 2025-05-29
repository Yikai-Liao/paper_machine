---
title: "Pretraining Language Models to Ponder in Continuous Space"
pubDatetime: 2025-05-27T03:47:33+00:00
slug: "2025-05-pondering-continuous-space"
type: "arxiv"
id: "2505.20674"
score: 0.9051415742015179
author: "grok-3-latest"
authors: ["Boyi Zeng", "Shixiang Song", "Siyuan Huang", "Yixuan Wang", "He Li", "Ziwei He", "Xinbing Wang", "Zhiyu Li", "Zhouhan Lin"]
tags: ["LLM", "Continuous Space", "Pre-Training", "Test Time Scaling", "Reasoning"]
institution: ["Shanghai Jiao Tong University", "Institute for Advanced Algorithms Research, Shanghai", "Shanghai Innovation Institute"]
description: "本文提出 Pondering Language Model，通过自监督学习在预训练阶段引入连续空间的迭代思考机制，显著提升语言模型性能并为测试时计算扩展提供新维度。"
---

> **Summary:** 本文提出 Pondering Language Model，通过自监督学习在预训练阶段引入连续空间的迭代思考机制，显著提升语言模型性能并为测试时计算扩展提供新维度。 

> **Keywords:** LLM, Continuous Space, Pre-Training, Test Time Scaling, Reasoning

**Authors:** Boyi Zeng, Shixiang Song, Siyuan Huang, Yixuan Wang, He Li, Ziwei He, Xinbing Wang, Zhiyu Li, Zhouhan Lin

**Institution(s):** Shanghai Jiao Tong University, Institute for Advanced Algorithms Research, Shanghai, Shanghai Innovation Institute


## Problem Background

大型语言模型（LLMs）通过增加参数和数据规模提升性能面临数据枯竭、计算成本高昂及性能饱和等问题，而测试时计算扩展（如链式思维 CoT）依赖人工标注数据或复杂强化学习，且局限于离散词汇空间，限制了内部计算思维能力；本文受人类反复思考（pondering）的启发，旨在通过自监督学习引入类似机制，提升模型推理能力并突破离散空间限制。

## Method

* **核心思想:** 提出 Pondering Language Model（Pondering LM），在单个 token 生成步骤内通过多次前向传播进行迭代‘思考’，生成连续的‘pondering embedding’来精炼预测分布，而非直接采样离散 token。
* **具体实现:** 
  * 模型根据当前预测概率分布，对词汇表中所有 token 的嵌入进行加权求和，生成一个连续的 pondering embedding。
  * 通过残差连接将 pondering embedding 与原始输入嵌入相加，形成更新后的输入嵌入。
  * 将更新后的嵌入反馈到模型中，进行下一次前向传播，重复此过程 k 步（pondering steps），最终基于精炼后的概率分布计算损失并优化模型。
  * 为提高效率，仅考虑概率最高的 top-K token 进行加权求和，减少计算复杂度。
* **训练方式:** 完全基于自监督学习，可直接集成到预训练阶段，无需人工标注数据或强化学习，适用于现有语言模型架构。
* **优势:** 突破离散词汇空间限制，允许模型在连续空间内进行更灵活的内部计算和推理，同时提升参数知识密度，减少大规模训练的通信成本。

## Experiment

* **有效性:** 实验覆盖 GPT-2、LLaMA 和 Pythia 架构，参数规模从 14M 到 1.4B，Pondering LM 在语言建模任务上困惑度显著降低，例如 PonderingPythia-1B 性能接近训练数据量多 10 倍的 TinyLlama-1.1B。
* **下游任务表现:** 在 9 个通用下游任务和指令跟随任务（MT-Bench）中，PonderingPythia 模型在零样本和少样本设置下均显著优于官方 Pythia 模型，显示出强泛化能力。
* **可扩展性:** 增加 pondering steps 数量持续降低语言建模损失，表明方法潜力巨大。
* **实验设置:** 从小规模验证到大规模预训练（Pile 数据集 300B tokens），对比充分，设置合理，但推理时计算开销随 pondering steps 线性增加，为潜在局限。

## Further Thoughts

连续空间的 pondering 机制为模型内部推理提供了更大自由度，是否可扩展至多模态模型（如视觉-语言模型）以增强跨领域推理能力？此外，token-adaptive pondering 的概念启发我们探索动态调整思考步骤以优化效率；同时，pondering embedding 的变化轨迹是否可用于可视化模型推理过程，从而提升模型解释性？