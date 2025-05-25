---
title: "Don't "Overthink" Passage Reranking: Is Reasoning Truly Necessary?"
pubDatetime: 2025-05-22T16:41:37+00:00
slug: "2025-05-reasoning-reranking-analysis"
type: "arxiv"
id: "2505.16886"
score: 0.8066425947926584
author: "grok-3-latest"
authors: ["Nour Jedidi", "Yung-Sung Chuang", "James Glass", "Jimmy Lin"]
tags: ["LLM", "Information Retrieval", "Reranking", "Reasoning", "Partial Relevance"]
institution: ["MIT Lincoln Laboratory", "Massachusetts Institute of Technology", "University of Waterloo"]
description: "本文通过对比实验揭示推理过程在点式重排序任务中限制了模型对部分相关性的建模能力，导致性能下降，建议使用更简单高效的标准重排序方法。"
---

> **Summary:** 本文通过对比实验揭示推理过程在点式重排序任务中限制了模型对部分相关性的建模能力，导致性能下降，建议使用更简单高效的标准重排序方法。 

> **Keywords:** LLM, Information Retrieval, Reranking, Reasoning, Partial Relevance

**Authors:** Nour Jedidi, Yung-Sung Chuang, James Glass, Jimmy Lin

**Institution(s):** MIT Lincoln Laboratory, Massachusetts Institute of Technology, University of Waterloo


## Problem Background

近年来，大型语言模型（LLMs）通过生成显式推理过程（Chain-of-Thought, CoT）在复杂自然语言任务中表现出色，这促使信息检索（IR）领域的研究者探索将推理能力整合到基于LLM的段落重排序（Passage Reranking）系统中。
然而，显式推理过程是否真正能提升重排序准确性仍未明确，论文聚焦于点式重排序器（Pointwise Rerankers），研究推理过程的必要性及其对性能的影响，特别是推理是否限制了模型对部分相关性（Partial Relevance）的建模能力。

## Method

*   **核心思想:** 对比推理式点式重排序器与标准无推理重排序器的性能，探究推理过程是否对重排序准确性有正面影响。
*   **具体实现:** 设计了三种重排序器进行对比：
    *   **StandardRR（标准重排序器）:** 直接基于查询-段落对，通过LLM输出相关性分数（‘true’或‘false’），不生成推理过程。训练时使用（查询，段落，相关性标签）三元组数据，推理时通过softmax计算‘true’的概率作为相关性分数R。
    *   **ReasonRR（推理式重排序器）:** 在输出相关性分数前，生成显式推理过程（Chain-of-Thought），以模拟人类推理方式辅助判断。训练时使用（查询，段落，推理过程，相关性标签）四元组数据，推理时基于生成的推理过程计算相关性分数R。
    *   **ReasonRR-NoReason（禁用推理的推理式重排序器）:** 基于ReasonRR，但在推理时通过预填充固定推理文本（如‘<think> Okay, I think I have finished thinking. </think>’）禁用推理过程，实质上将其转化为标准点式重排序器。
*   **训练与评估细节:** 三种方法在相同的训练数据（基于MS MARCO增强的Rank1数据集）和骨干模型（Qwen2.5系列，参数规模1.5B至7B）下，使用LoRA微调进行训练。评估指标为NDCG@10，数据集包括领域内（MS MARCO系列）和领域外（BRIGHT，推理密集型检索基准）。
*   **关键点:** 通过控制变量（训练数据、模型规模、微调方法）确保对比公平，重点分析推理过程对相关性分数分布和重排序效果的影响。

## Experiment

*   **有效性对比:** 实验结果表明，StandardRR在领域内（MS MARCO）和领域外（BRIGHT）数据集上平均性能优于ReasonRR。例如，在MS MARCO上，StandardRR在1.5B、3B、7B模型规模下的NDCG@10分别比ReasonRR高5.3、3.7和5个百分点；在BRIGHT上，差距分别为3.4、1和3.2个百分点。
*   **推理过程影响:** 禁用推理过程后，ReasonRR-NoReason在MS MARCO上平均提升了0.8、0.5和1.4个百分点（对应1.5B、3B、7B规模），在BRIGHT上7B规模模型提升了3个百分点，表明推理过程对ReasonRR性能有负面影响。
*   **原因分析:** 推理过程导致ReasonRR的相关性分数分布极化（Polarized），倾向于输出极高或极低分数，忽视部分相关性（Partial Relevance），而StandardRR和ReasonRR-NoReason能更好捕捉部分相关性，从而提升重排序准确性。
*   **实验设置合理性:** 实验覆盖不同模型规模（1.5B至7B）、领域内和领域外数据集，使用标准指标NDCG@10，训练和推理细节（如LoRA微调、Pyserini和vLLM实现）规范。不过，仅限于Qwen2.5模型家族和7B以下规模，未覆盖更大模型或不同模型家族，存在一定局限性。
*   **改进尝试:** 论文尝试通过自一致性（Self-Consistency）方法改进ReasonRR，将多个推理输出的相关性分数取平均，NDCG@10在MS MARCO和BRIGHT上分别提升1.8和2.9个百分点，但仍不及StandardRR。

## Further Thoughts

论文揭示推理过程在某些任务中可能并非必要，甚至可能成为性能瓶颈，这一发现启发我们在设计LLM应用时应审视推理机制是否适合特定任务。例如，在重排序任务中，显式推理可能引入不必要的复杂性，而直接优化相关性分数分布可能更有效。此外，部分相关性建模问题的提出，启发我们可以在其他任务中探索非二元标签的训练方法，或设计专门的损失函数来校准模型输出分布，特别是在需要细粒度判断的场景中。