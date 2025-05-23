---
title: "Text Generation Beyond Discrete Token Sampling"
pubDatetime: 2025-05-20T18:41:46+00:00
slug: "2025-05-mixture-inputs-generation"
type: "arxiv"
id: "2505.14827"
score: 0.5665194967736968
author: "grok-3-latest"
authors: ["Yufan Zhuang", "Liyuan Liu", "Chandan Singh", "Jingbo Shang", "Jianfeng Gao"]
tags: ["LLM", "Text Generation", "Embedding Space", "Bayesian Inference", "Reasoning"]
institution: ["UC San Diego", "Microsoft Research"]
description: "本文提出 Mixture of Inputs (MOI) 方法，通过贝叶斯估计将 token 分布信息融入自回归生成输入，无需训练即可显著提升大型语言模型在推理任务上的表现。"
---

> **Summary:** 本文提出 Mixture of Inputs (MOI) 方法，通过贝叶斯估计将 token 分布信息融入自回归生成输入，无需训练即可显著提升大型语言模型在推理任务上的表现。 

> **Keywords:** LLM, Text Generation, Embedding Space, Bayesian Inference, Reasoning

**Authors:** Yufan Zhuang, Liyuan Liu, Chandan Singh, Jingbo Shang, Jianfeng Gao

**Institution(s):** UC San Diego, Microsoft Research


## Problem Background

大型语言模型（LLMs）在自回归生成中通常预测下一个 token 的概率分布并采样离散 token 作为输出，而丢弃了分布中的丰富信息，导致模型只能沿单一路径推理，限制了其在复杂任务中的表现。
作者受人类思维高维、流动特性的启发，提出保留分布信息以增强模型内部表示，从而提升文本质量和推理能力。

## Method

*   **核心思想:** 提出 Mixture of Inputs (MOI)，一种无需训练的自回归生成改进方法，通过将采样得到的离散 token 和概率分布结合，形成混合输入，以保留模型的不确定性和多种可能性。
*   **具体实现:** 
    *   在每个解码步骤，模型输出下一个 token 的概率分布，并通过现有采样策略（如 top-k 或温度缩放）选择一个离散 token 作为输出。
    *   使用贝叶斯估计方法，将概率分布视为先验（prior），采样 token 视为观测（observation），计算后验期望（posterior expectation）作为新的输入表示。
    *   后验期望通过加权平均的方式，将分布中多个可能 token 的嵌入（embedding）线性组合为一个连续的混合嵌入向量，替代传统的 one-hot 向量输入到下一解码步骤。
    *   引入超参数 β 控制分布信息和采样 token 的相对权重，β 值影响混合输入中不确定性和确定性信息的平衡，需根据任务特性调整。
*   **优势:** MOI 不改变模型架构，与现有解码策略兼容，计算开销极低，可直接应用于现有模型，无需额外训练或微调。

## Experiment

*   **有效性:** MOI 在四个推理密集型任务（AIME, Count Down 4, GPQA-Diamond, LiveCodeBench）和四个模型（QwQ-32B, Nemotron-Super-49B, Gemma-3-27B, DAPO-Qwen-32B）上测试，16 个模型-任务对中均匹配或优于标准基线，平均绝对提升 1.8%，如 Nemotron-Super-49B 在 GPQA-Diamond 上提升 4.05%。
*   **任务特异性:** 在需要多步符号操作的任务（如 Count Down 4）上提升最显著（平均提升 3.7%），表明保留不确定性有助于减少多步推理中的累积错误；其他任务如 AIME 和 GPQA 也有稳定提升。
*   **对比分析:** 相比直接使用分布作为输入的消融方法（Direct Mixture），MOI 的贝叶斯平滑机制至关重要，直接混合往往导致性能下降（如 Nemotron-Super-49B 在 LiveCodeBench 上下降 22.9%）。
*   **计算开销:** MOI 的计算开销极低，在 vLLM 框架下输入和输出吞吐量分别仅下降 2.4% 和 3.66%。
*   **实验设置合理性:** 实验覆盖多种任务和模型规模，设置标准基线和消融实验，通过网格搜索优化超参数（如 β, top-p, 温度），报告多次运行平均值，结果稳健；但未涉及开放性生成任务，存在一定局限性。

## Further Thoughts

MOI 利用分布信息进行‘内省式推理’的思路启发我们，是否可以在提示设计或中间状态处理中引入类似混合机制以优化复杂任务理解；
超参数 β 的任务依赖性提示是否可以设计自适应调整策略，在推理时动态优化；
嵌入空间线性操作的成功是否意味着可以探索非线性组合方式或直接在连续空间中设计模型，更加贴近人类思维模式。