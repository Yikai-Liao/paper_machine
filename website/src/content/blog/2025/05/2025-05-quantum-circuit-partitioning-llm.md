---
title: "Circuit Partitioning Using Large Language Models for Quantum Compilation and Simulations"
pubDatetime: 2025-05-12T16:18:48+00:00
slug: "2025-05-quantum-circuit-partitioning-llm"
type: "arxiv"
id: "2505.07711"
score: 0.651966809904151
author: "grok-3-latest"
authors: ["Pranav Sinha", "Sumit Kumar Jha", "Sunny Raj"]
tags: ["LLM", "Quantum Computing", "Circuit Partitioning", "Fine-Tuning", "Code Generation"]
institution: ["Oakland University", "Florida International University"]
description: "本文提出通过低秩适应（LoRA）微调大型语言模型（LLMs）以模拟 quick partition 算法进行量子电路分区，取得了 53.39% 的准确率并保证 100% 代码等价性，展示了 LLMs 在量子计算领域的应用潜力。"
---

> **Summary:** 本文提出通过低秩适应（LoRA）微调大型语言模型（LLMs）以模拟 quick partition 算法进行量子电路分区，取得了 53.39% 的准确率并保证 100% 代码等价性，展示了 LLMs 在量子计算领域的应用潜力。 

> **Keywords:** LLM, Quantum Computing, Circuit Partitioning, Fine-Tuning, Code Generation

**Authors:** Pranav Sinha, Sumit Kumar Jha, Sunny Raj

**Institution(s):** Oakland University, Florida International University


## Problem Background

在嘈杂中等规模量子（NISQ）时代，量子计算机受限于噪声门（如 CNOT 门）的影响，导致计算结果不可靠；量子电路编译旨在将高层次算法映射到硬件支持的基本门上并减少噪声门，但由于计算复杂性，现有算法无法处理超过 5-6 个量子比特的电路，因此需要先对大电路进行分区以便后续优化；现有分区方法多为启发式，缺乏对下游任务的针对性优化，论文探索是否能利用大型语言模型（LLMs）通过学习现有分区算法（如 quick partition）实现更智能的分区。

## Method

* **核心思想**：利用大型语言模型（LLMs）的代码理解与生成能力，通过微调使其学习量子电路分区任务，模拟 Berkeley Quantum Synthesis Toolkit (BQSKit) 中的 quick partition 方法。
* **数据准备**：从 Munich Quantum Toolkit Benchmark Library (MQT Bench) 获取量子电路及其分区结果，清理代码以减少 token 数量（如移除注释、测量指令），并将浮点数替换为符号以降低处理难度。
* **提示设计**：设计特定提示（如‘Create barriers for efficient processing’），指导 LLMs 在分区时保持多量子比特门的依赖顺序，避免错误重排。
* **学习目标与算法**：以 quick partition 为基准，这是一种简单的一遍式分区方法，按门执行顺序分配到活动分区中，并通过屏障（barrier）分隔不同块；LLMs 被训练以生成类似的分区结果。
* **训练策略**：由于零样本和少样本学习效果不佳，采用低秩适应（Low-Rank Adaptation, LoRA）方法进行微调，通过冻结原始模型权重，仅更新低秩分解矩阵，降低内存需求和训练成本。
* **模型选择**：测试多个开源 LLMs，包括 Llama-3.1 系列（8B 和 70B）、Mistral-7B、CodeLlama-7B 和 Phi-3-mini-128k-instruct，探索不同模型架构和参数规模对分区任务的影响。

## Experiment

* **有效性**：通过 LoRA 微调，Llama-3.1-70B 模型在分区任务上取得了 53.39% 的准确率（即与 quick partition 结果完全一致的比例），Llama-3.1-8B 达到 51.69%；即使分区不完全一致，生成的代码在 100% 的情况下与原始电路等价，表明模型不会破坏电路功能。
* **对比分析**：少样本学习（1-shot 和 5-shot）完全失效，准确率为 0%，模型倾向于重复输入代码，而微调显著提升了性能。
* **模型规模影响**：参数规模更大的模型（如 Llama-3.1-70B）表现更好，但训练时间显著增加（每轮约 5 小时）；小型模型（如 Phi-3-mini-128k-instruct）准确率最低，仅 15.45%。
* **实验设置合理性**：实验在 8 个 Nvidia H100 GPU 上进行，使用 MQT Bench 数据集（token 限制在 6000 以内），80% 用于训练，20% 用于测试，覆盖多种模型和训练策略；但上下文长度限制导致部分大电路数据无法使用，可能影响泛化能力；此外，未探讨微调后模型是否能超越 quick partition 或优化下游任务。

## Further Thoughts

论文展示了 LLMs 在量子计算等专业领域的潜力，启发我们思考是否可将其应用于其他需要模式识别和代码生成的计算任务；上下文长度限制提示未来可探索支持更大上下文的模型或分层处理策略；微调结合下游任务（如减少 CNOT 门）的想法启发我们通过多任务学习或强化学习直接优化量子电路噪声性能；数据预处理（如 token 优化）的重要性提示我们在其他领域应用 LLMs 时，数据表示优化可能是关键突破点。