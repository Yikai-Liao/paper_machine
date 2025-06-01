---
title: "From Token to Action: State Machine Reasoning to Mitigate Overthinking in Information Retrieval"
pubDatetime: 2025-05-29T04:04:25+00:00
slug: "2025-05-state-machine-reasoning"
type: "arxiv"
id: "2505.23059"
score: 0.52331504936927
author: "grok-3-latest"
authors: ["Dohyeon Lee", "Yeonseok Jeong", "Seung-won Hwang"]
tags: ["LLM", "Information Retrieval", "Reasoning", "State Machine", "Token Efficiency"]
institution: ["Seoul National University"]
description: "本文提出状态机推理（SMR）框架，通过结构化状态转换和信息检索特定动作有效缓解链式思维中的过度思考问题，显著提升检索性能并减少 token 使用量。"
---

> **Summary:** 本文提出状态机推理（SMR）框架，通过结构化状态转换和信息检索特定动作有效缓解链式思维中的过度思考问题，显著提升检索性能并减少 token 使用量。 

> **Keywords:** LLM, Information Retrieval, Reasoning, State Machine, Token Efficiency

**Authors:** Dohyeon Lee, Yeonseok Jeong, Seung-won Hwang

**Institution(s):** Seoul National University


## Problem Background

大型语言模型（LLMs）在信息检索（IR）任务中应用链式思维（Chain-of-Thought, CoT）提示时，常常因生成冗长且语义冗余的推理轨迹而导致‘过度思考’（Overthinking），不仅效率低下，还可能偏离用户意图。
论文识别出两个关键问题：一是冗余轨迹（Redundant Trajectories），即模型在 token 级别反复访问语义相似的状态；二是误导推理（Misguided Reasoning），即通过强化学习压缩推理轨迹可能导致输出与用户意图不符，尤其在开放域任务中。

## Method

*   **核心思想**：提出状态机推理（State Machine Reasoning, SMR）框架，将推理过程建模为结构化状态之间的离散转换，而非传统的 token 级生成，以避免冗余和误导推理。
*   **状态表示**：每个推理步骤的状态用一个结构化元组 (q, D) 表示，其中 q 是当前查询，D 是检索到的文档列表；这种显式表示便于检测语义重复，避免不必要的推理循环。
*   **动作空间**：定义了三个针对 IR 任务的离散动作：REFINE（查询重写，通过更新查询以更好地反映用户需求）、RERANK（文档重排序，调整文档列表顺序以提升相关性）和 STOP（终止推理，当检测到状态等价或无进一步改进时停止）。
*   **策略选择**：采用基于提示的策略模型，利用 LLM 作为决策者，根据当前状态动态选择动作，而非依赖固定启发式或训练控制器；提示中嵌入规则引导 LLM 决策，例如当查询模糊或结果不满意时选择 REFINE。
*   **停止机制**：通过检测状态是否等价（即查询和文档列表无显著变化）实现早期停止，同时设置最大步骤数（如 16 步）以控制计算成本，确保推理过程的高效性。
*   **优势**：SMR 通过结构化状态和动作控制推理，避免了 token 级生成的盲目性和语义漂移，同时保持与用户意图的对齐，且无需任务特定训练即可泛化到不同模型和检索器。

## Experiment

*   **有效性**：在 BRIGHT 基准数据集上，SMR 的 nDCG@10 指标平均提升了 3.4%，与最强基线相比，在稀疏检索器（BM25）下提升 5.4%，在密集检索器（ReasonIR）下提升 2.1%；在 BEIR 基准数据集上，SMR 平均 nDCG@10 达到 49.8%，同样优于标准 CoT 和压缩 CoT 方法，表明其在复杂和标准 IR 任务中均有效。
*   **效率提升**：SMR 的 token 使用量减少了 74.4%，远超基线方法（如 O1-Pruner 仅减少 5%），得益于早期停止机制和结构化状态管理，避免了冗余推理。
*   **实验设置合理性**：实验覆盖了多种 LLM（如 Qwen2.5-32B 和 QwQ-32B）和检索器（稀疏 BM25 和密集 ReasonIR），并与标准 CoT（如 Rank1、Rank-R1）和压缩 CoT（如 O1-Pruner）进行了对比；数据集选择兼顾推理复杂度和通用性，实验设计全面。
*   **额外验证**：通过动作分布分析，SMR 展现出适应性强的策略选择，例如在初始检索结果较差的领域更多选择 REFINE 动作；意图对齐分数始终保持在 0.9 以上，表明查询重写未偏离用户意图；消融研究进一步验证了提示策略的有效性。

## Further Thoughts

SMR 将状态机概念从数学和代码验证领域扩展到信息检索的创新尝试启发了我，状态机的结构化推理方式或许可以应用到其他需要控制推理轨迹的 NLP 任务中，如对话系统或多步决策；此外，通过定义离散动作约束推理过程的思路提示我们，可以设计更多任务特定的动作空间，例如在问答任务中加入‘证据提取’动作；最后，SMR 不依赖任务特定训练而通过提示实现通用性的设计，启发未来推理框架可以更多依赖提示工程而非模型微调，以降低开发成本并提升跨任务适应性。