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
description: "CASTILLO 数据集通过表征 13 个开源大型语言模型在 7 个指令跟随任务上的响应长度分布，为推理效率优化和系统调度提供了关键数据支持。"
---

> **Summary:** CASTILLO 数据集通过表征 13 个开源大型语言模型在 7 个指令跟随任务上的响应长度分布，为推理效率优化和系统调度提供了关键数据支持。 

> **Keywords:** LLM, Response Length, Variability, Scheduling, Inference Optimization

**Authors:** Daniel F. Perez-Ramirez, Dejan Kostic, Magnus Boman

**Institution(s):** KTH Royal Institute of Technology, RISE Computer Science, Karolinska Institutet


## Problem Background

大型语言模型（LLMs）在自回归文本生成过程中表现出随机性和变异性，导致响应长度难以预测，这给生产系统中的计算和内存资源管理带来了挑战，尤其是在高并发和低延迟需求的场景下。
现有方法要么对生成过程引入长度偏见，要么基于忽略模型和提示特定变异性的简化假设，因此需要系统性数据来支持响应长度预测和主动资源调度。

## Method

* **数据集构建**：CASTILLO 数据集涵盖了 13 个开源大型语言模型（包括 LLaMA、Mistral、Qwen 等家族，参数规模从 1B 到 70B）和 7 个指令跟随数据集（包括 Dolly、ShareGPT、Alpaca 等，覆盖通用 NLP 和代码生成任务），以捕捉多样化的生成行为。
* **响应生成流程**：对每个‘提示-模型’对，使用固定的解码超参数（如 temperature、top-k、top-p），生成 10 个独立响应，记录每个响应的 token 长度，并限制输入提示长度为 2500 token，输出长度上限为 15000 token，以平衡覆盖率和计算成本。
* **统计与分析**：计算响应长度的均值、标准差、百分位数（P25, P50, P75, P99）等指标，保存最短和最长响应文本及生成设置，通过热图、箱线图和变异系数（CV）分析模型间和模型内的长度变异性。
* **退化处理**：采用两阶段过滤策略（基于输出长度上限和标准差阈值）识别文本退化样本，将数据集分为‘干净’版本和‘退化’子集，支持退化现象的独立研究。
* **扩展性设计**：数据集和代码支持添加新模型、数据集和解码配置，并可选缓存预填充阶段的激活值（hidden states 和 logits），为未来分析输入表征与输出长度的关系提供可能。

## Experiment

* **变异性揭示**：实验表明响应长度存在显著的模型间和模型内变异性，例如不同模型对相同提示的均值响应长度差异可达数百 token，同一模型内 10 个响应的变异系数高达 45%，凸显了预测难度。
* **退化现象**：识别出 956 个退化样本，某些模型（如 llama-1B）占退化案例的 40% 以上，较小模型比大模型更易出现退化，数据集（如 ShareGPT 和 Apps）因提示复杂性也贡献了较多退化案例。
* **设置全面性**：实验覆盖多种模型架构、参数规模和任务类型，通过分层采样确保提示分布代表性，多次独立生成（10 次）捕捉随机性，设计合理。
* **局限性**：实验仅限于 Transformer 架构，未系统研究解码参数的影响，且未涉及在线生成场景的实时性测试。

## Further Thoughts

CASTILLO 数据集为响应长度预测模型的开发提供了基础，未来可以探索将预测模型嵌入到推理系统中，实现动态资源分配；此外，模型特定行为和退化现象的差异提示我们可以通过定制化训练或解码策略优化特定模型的生成稳定性；另外，数据集扩展到链式推理（Chain-of-Thought）场景的想法非常有前景，可以分别预测推理过程和最终答案的长度，为复杂任务的资源管理提供更精细的支持。