---
title: "CASTILLO: Characterizing Response Length Distributions of Large Language Models"
pubDatetime: 2025-05-22T16:35:33+00:00
slug: "2025-05-castillo-response-length"
type: "arxiv"
id: "2505.16881"
score: 0.7289506014626772
author: "grok-3-latest"
authors: ["Daniel F. Perez-Ramirez", "Dejan Kostic", "Magnus Boman"]
tags: ["LLM", "Response Length", "Variability", "Scheduling", "Inference Optimization"]
institution: ["KTH Royal Institute of Technology", "RISE Computer Science", "Karolinska Institutet"]
description: "本文通过构建 CASTILLO 数据集，系统表征了 13 个开源大型语言模型在 7 个指令跟随数据集上的响应长度分布，为预测模型和主动推理调度提供了关键资源。"
---

> **Summary:** 本文通过构建 CASTILLO 数据集，系统表征了 13 个开源大型语言模型在 7 个指令跟随数据集上的响应长度分布，为预测模型和主动推理调度提供了关键资源。 

> **Keywords:** LLM, Response Length, Variability, Scheduling, Inference Optimization

**Authors:** Daniel F. Perez-Ramirez, Dejan Kostic, Magnus Boman

**Institution(s):** KTH Royal Institute of Technology, RISE Computer Science, Karolinska Institutet


## Problem Background

大型语言模型（LLM）在生产环境中的推理面临高计算和内存需求，尤其在高并发和低延迟约束下，资源管理效率低下。
由于自回归文本生成长度的随机性和可变性，现有资源分配方法难以准确预测响应长度，导致资源利用率低和运营成本高。
论文旨在通过表征响应长度分布，解决生成前预测长度的问题，以支持主动资源调度。

## Method

*   **核心目标:** 构建一个大规模数据集 CASTILLO，用于表征开源大型语言模型（LLM）在不同任务上的响应长度分布，以支持长度预测和系统优化。
*   **数据集构建:** 选择了 7 个公开指令跟随数据集（包括 Dolly、ShareGPT、Alpaca 等通用 NLP 数据集和 Mbpp、Apps 等代码生成数据集），确保任务类型和语料多样性，部分数据集通过分层采样限制为 2000 个样本。
*   **模型选择:** 涵盖 13 个开源指令微调 LLM，来自 LLaMA、Mistral、Qwen、Phi 和 Gemma 家族，参数规模从 1B 到 70B，允许模型间和模型家族内的比较。
*   **响应生成流程:** 对每个提示-模型对，使用固定解码超参数（如 temperature、top-k、top-p）生成 10 个独立完成，记录每个响应的 token 长度，并存储统计数据（均值、标准差、百分位数）以及最短和最长完成文本。
*   **数据处理与限制:** 设置输入提示长度上限为 2500 token，输出长度上限为 15000 token，以控制 GPU 内存使用和避免病态生成；同时通过两阶段过滤策略识别文本退化（degeneration），将数据集分为 sanitized（无退化）和 degeneration-only（仅退化）子集。
*   **技术实现:** 使用 Hugging Face 的 Transformers 库生成响应，在 HPC 集群上利用 Nvidia H100 GPU 进行计算，记录详细生成配置以支持可重复性。

## Experiment

*   **响应长度变异性:** 实验揭示了显著的模型间变异性（不同模型对相同提示的响应长度差异可达数百 token）和模型内变异性（同一模型对不同提示或同一提示多次生成的响应长度方差高），表明预测难度的复杂性。
*   **数据集与模型特异性:** 代码相关数据集（如 Apps、DS-1000）通常引发更长响应，而文本数据集（如 Alpaca）响应较短；但某些模型（如 Gemma）表现出相反趋势，显示模型架构和调优对生成行为的影响。
*   **文本退化现象:** 识别出 956 个文本退化样本（表现为重复或不连贯），小型模型（如 llama-1B）退化率较高，占总退化案例的 40% 以上，而部分大型模型（如 minist-8B）无退化，表明模型规模与生成稳定性相关。
*   **实验设置评价:** 实验覆盖了多种模型（1B-70B 参数）和数据集，生成多次独立完成以捕捉随机性，设置长度限制避免资源耗尽，并提供退化过滤机制，设计较为全面合理；但局限在于仅关注 Transformer 架构，未涉及其他模型类型。
*   **数据可用性:** 数据集和代码已公开发布，支持后续研究和系统模拟。

## Further Thoughts

CASTILLO 数据集揭示的模型特异性和响应变异性启发我们，不仅可以基于此开发通用长度预测模型，还可能针对不同模型定制调度策略，甚至在训练阶段引入长度约束机制；此外，文本退化现象的模型和数据集依赖性提示我们探索解码参数对生成稳定性的影响，或设计实时退化检测算法以提升推理质量；最后，论文提到的链式推理（Chain-of-Thought）长度预测思路可进一步扩展到多阶段任务中，优化复杂推理场景的资源分配。