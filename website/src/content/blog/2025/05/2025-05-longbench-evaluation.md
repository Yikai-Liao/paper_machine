---
title: "100-LongBench: Are de facto Long-Context Benchmarks Literally Evaluating Long-Context Ability?"
pubDatetime: 2025-05-25T19:58:31+00:00
slug: "2025-05-longbench-evaluation"
type: "arxiv"
id: "2505.19293"
score: 0.6460496006875236
author: "grok-3-latest"
authors: ["Wang Yang", "Hongye Jin", "Shaochen Zhong", "Song Jiang", "Qifan Wang", "Vipin Chaudhary", "Xiaotian Han"]
tags: ["LLM", "Long Context", "Evaluation Metrics", "Benchmark Design", "Context Length"]
institution: ["Case Western Reserve University", "Texas A&M University", "Rice University", "Meta"]
description: "本文提出长度可控的长上下文基准 -LongBench 和新指标 LongScore，通过隔离基础能力显著提升了 LLMs 长上下文能力的评估精度。"
---

> **Summary:** 本文提出长度可控的长上下文基准 -LongBench 和新指标 LongScore，通过隔离基础能力显著提升了 LLMs 长上下文能力的评估精度。 

> **Keywords:** LLM, Long Context, Evaluation Metrics, Benchmark Design, Context Length

**Authors:** Wang Yang, Hongye Jin, Shaochen Zhong, Song Jiang, Qifan Wang, Vipin Chaudhary, Xiaotian Han

**Institution(s):** Case Western Reserve University, Texas A&M University, Rice University, Meta


## Problem Background

大型语言模型（LLMs）的长上下文能力被认为是其核心竞争力之一，能够帮助用户处理长篇文档理解等复杂任务。然而，现有长上下文评估基准（如 LongBench、L-Eval）存在两大缺陷：一是未能有效区分模型的基础能力（Base Ability）和长上下文能力，导致评估结果混淆；二是数据样本长度固定，无法适应不同上下文窗口大小的模型，也无法测试模型在不同长度下的性能衰减点。因此，亟需一个更精准、灵活的评估框架来揭示 LLMs 在长上下文场景下的真实表现。

## Method

* **核心创新点：** 提出一个长度可控的长上下文基准 -LongBench 和一个新评估指标 LongScore，旨在更准确地评估 LLMs 的长上下文能力。
* **-LongBench 设计：** 
  * 包含多种任务类型（如关键信息检索、信息理解、单文档和多文档问答、总结等），任务难度分级，覆盖真实和合成数据，模拟现实场景。
  * 数据生成过程：从真实上下文来源（Real Context Sources）和噪声上下文来源（Noisy Context Sources）中抽取文章，构建指定长度的上下文（如 128k 令牌），并通过随机打乱文章顺序增加复杂性。
  * 引入 QA 过滤机制：在问答任务中，通过无上下文场景测试模型，若模型依赖先验知识（得分超过阈值），则排除该数据，以减少先验知识对评估的干扰。
* **LongScore 指标：** 
  * 目标是隔离模型的基础能力（Base Ability）和长上下文能力，通过计算模型在长上下文长度下的性能与基础能力的相对差异来评估长上下文能力。
  * 具体计算：首先基于短上下文长度（2k、4k、6k）的平均性能定义 Base Ability；然后对于长上下文长度（如 8k、16k、32k 等），计算 LongScore = S_l - Base Ability，其中 S_l 是模型在长度 l 下的性能得分。
  * 优势：避免基础能力对长上下文评估的干扰，提供更精准的排名和性能差异分析。
* **实现细节：** 任务评估结合了准确率（Accuracy）和基于 LLM 的评分（如流畅性和正确性），确保评估的多维度性和可靠性。

## Experiment

* **可靠性验证：** 通过测试同一模型家族中不同规模的模型（如 Llama 3.1、Qwen 2.5），观察到大模型通常表现更好，且性能随上下文长度增加而下降，符合预期趋势，证明 -LongBench 设计合理。
* **有效性验证：** LongScore 指标在区分长上下文能力方面优于传统平均分指标。例如，在 NTK 和 PI 方法的对比中，LongScore 放大了性能差异（NTK 的 LongScore 差距更大），揭示了 NTK 在长上下文上的优势，而传统指标未能清晰区分。
* **全面性与合理性：** 实验覆盖多个开源模型（如 Llama 3.1、Qwen 2.5、Phi 3），测试上下文长度从 8k 到 256k，包含八种任务类型，并引入医疗和法律领域特定任务，设置全面且贴近实际应用场景。
* **显著性：** LongScore 改变了模型排名，例如 Llama 3.1-8B 在长上下文（如 128k、256k）上表现优于 Qwen 2.5-7B，尽管后者在短上下文上得分更高，表明新指标更准确反映长上下文能力。
* **局限性：** 实验指出，若模型基础能力较弱，LongScore 评估可能波动较大；此外，构建短上下文任务时需收集多样化长度的文本，增加了数据准备成本。

## Further Thoughts

论文提出的长度可控基准设计和隔离基础能力的评估思路具有广泛适用性，例如可以扩展到多模态模型的评估，测试其在长序列图像或视频理解上的能力；此外，引入领域特定任务的做法启发我们在模型评估中应关注专业场景需求，可能揭示模型在特定领域的局限性；最后，LongScore 的相对差异计算方法提示我们，未来可以探索更多基于相对性能的评估指标，以减少无关变量对评估的干扰。