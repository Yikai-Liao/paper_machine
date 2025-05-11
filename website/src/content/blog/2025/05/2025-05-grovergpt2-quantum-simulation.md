---
title: "GroverGPT-2: Simulating Grover's Algorithm via Chain-of-Thought Reasoning and Quantum-Native Tokenization"
pubDatetime: 2025-05-08T01:38:12+00:00
slug: "2025-05-grovergpt2-quantum-simulation"
type: "arxiv"
id: "2505.04880"
score: 0.521631880115876
author: "grok-3-latest"
authors: ["Min Chen", "Jinglei Cheng", "Pingzhi Li", "Haoran Wang", "Tianlong Chen", "Junyu Liu"]
tags: ["LLM", "Quantum Simulation", "Tokenization", "Reasoning", "Scalability"]
institution: ["University of Pittsburgh", "University of North Carolina at Chapel Hill"]
description: "本文提出GroverGPT-2，通过量子原生分词和思维链推理，利用大型语言模型高效模拟Grover量子搜索算法，展示了经典机器内化量子逻辑的潜力，为探索经典与量子计算边界提供了新工具。"
---

> **Summary:** 本文提出GroverGPT-2，通过量子原生分词和思维链推理，利用大型语言模型高效模拟Grover量子搜索算法，展示了经典机器内化量子逻辑的潜力，为探索经典与量子计算边界提供了新工具。 

> **Keywords:** LLM, Quantum Simulation, Tokenization, Reasoning, Scalability

**Authors:** Min Chen, Jinglei Cheng, Pingzhi Li, Haoran Wang, Tianlong Chen, Junyu Liu

**Institution(s):** University of Pittsburgh, University of North Carolina at Chapel Hill


## Problem Background

量子计算在理论上对特定任务（如Grover搜索算法）具有显著优势，但实际量子优势的边界仍未明确。
经典模拟量子算法是探索这一边界的重要手段，然而面临计算成本和内存消耗的指数级增长挑战。
本文聚焦于一个更深层次的问题：经典机器（特别是大型语言模型，LLMs）是否不仅能模拟量子算法，还能理解和内化其逻辑，从而为研究经典与量子计算的分离提供新视角。

## Method

*   **核心思想:** 提出GroverGPT-2，一种基于大型语言模型（LLM）的框架，通过自然语言处理技术模拟Grover量子搜索算法，旨在捕捉量子电路的逻辑结构并生成可解释的推理过程。
*   **量子原生分词（Quantum-Native Tokenization）:** 设计了一种专门针对量子汇编语言（QASM）的分词策略，通过规则解析将量子电路表示（如门操作、量子比特标识）转化为紧凑的语义单元，显著减少输入序列长度，提高内存和计算效率。
*   **思维链推理训练（Chain-of-Thought, CoT）:** 采用监督微调（Supervised Fine-Tuning, SFT）方法，训练模型生成中间推理步骤，模拟量子算法的逐步逻辑过程，而非直接输出最终结果，增强模型的可解释性和对复杂任务的处理能力。
*   **参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）:** 使用低秩适应（LoRA）技术，仅更新模型的一小部分参数（针对注意力模块的查询和值矩阵），在保持性能的同时大幅降低计算成本，支持在消费级硬件上进行领域适配。
*   **输入处理与输出生成:** 模型接受纯QASM描述作为输入（无需额外自然语言提示），通过CoT推理输出包括中间步骤、标记状态和概率幅度的结构化文本，直接从电路表示中提取关键实体（如Oracle）并推理标记状态。
*   **关键创新:** 通过结合量子原生分词和CoT推理，GroverGPT-2不仅实现了高效模拟，还展示了经典机器学习和内化量子算法逻辑的潜力。

## Experiment

*   **有效性:** 在训练范围内（2-7量子比特），GroverGPT-2在搜索准确率（Searching Accuracy, SA）和保真度（Fidelity）上接近1.0，显著优于基线LLM模型（如DeepSeek-R1），表明其在模拟精度上的明显提升。
*   **泛化能力:** 在超出训练范围的量子比特数（8-9量子比特，甚至到13量子比特）上，模型仍保持较高性能（SA和Fidelity在0.9以上），但在更大规模（14-20量子比特）时性能逐渐下降，显示出泛化能力的局限。
*   **效率与扩展性:** 量子原生分词将输入序列压缩比从1.35提升至1.40，CoT推理长度比基线模型短数十倍，执行时间增长接近亚线性，远低于传统经典模拟方法（如状态向量、密度矩阵模拟）的指数级增长。
*   **实验设置合理性:** 实验覆盖了不同输入类型（完整电路和仅Oracle输入）、量子比特数（2-20）和标记状态数（1-3），设置较为全面；但在超大规模量子比特上的测试数据较少，且未充分探讨复杂量子算法的表现，可能存在一定局限性。

## Further Thoughts

GroverGPT-2展示了LLM通过适当的分词和推理训练，能够理解和模拟结构化的量子电路逻辑，这启发我们思考：是否可以将LLM的推理能力推广到其他科学计算领域（如物理模拟或化学建模），通过‘学习而非计算’的范式避免传统方法的计算瓶颈？此外，结合符号推理与数值计算，或将LLM与传统量子模拟器集成，是否能进一步提升模拟精度和规模？