---
title: "Arbitrarily Applicable Same/Opposite Relational Responding with NARS"
pubDatetime: 2025-05-11T18:03:37+00:00
slug: "2025-05-relational-responding-nars"
type: "arxiv"
id: "2505.07079"
score: 0.4762141283435002
author: "grok-3-latest"
authors: ["Robert Johansson", "Patrick Hammer", "Tony Lofthouse"]
tags: ["AGI", "Relational Learning", "Symbolic Reasoning", "Contextual Cues", "Generalization"]
institution: ["Stockholm University", "KTH Royal Institute of Technology"]
description: "本文在非公理推理系统（NARS）中通过‘获取关系’机制，首次实现了任意适用的‘相同/相反’关系推导，结合对称性和传递性，为通用人工智能提供了接近人类符号推理的计算模型。"
---

> **Summary:** 本文在非公理推理系统（NARS）中通过‘获取关系’机制，首次实现了任意适用的‘相同/相反’关系推导，结合对称性和传递性，为通用人工智能提供了接近人类符号推理的计算模型。 

> **Keywords:** AGI, Relational Learning, Symbolic Reasoning, Contextual Cues, Generalization

**Authors:** Robert Johansson, Patrick Hammer, Tony Lofthouse

**Institution(s):** Stockholm University, KTH Royal Institute of Technology


## Problem Background

人类符号认知中的任意适用关系反应（Arbitrarily Applicable Relational Responding, AARR）是语言和抽象推理的基础，表现为基于任意上下文线索灵活泛化刺激关系的能力（如‘相同’和‘相反’）。
然而，计算建模这种能力对通用人工智能（AGI）是一个挑战，因为它需要在极少显式训练的基础上动态形成和操作关系结构，特别是在对称性（Mutual Entailment）和传递性（Combinatorial Entailment）方面的推导。

## Method

*   **核心思想:** 在非公理推理系统（NARS）中引入‘获取关系’（Acquired Relations）机制，通过感官-运动交互自适应学习和泛化关系结构，模拟人类符号认知中的上下文驱动关系推理。
*   **具体实现:** 
    *   使用匹配样本任务（Matching-to-Sample, MTS）作为实验范式，通过 NARS 的形式语言 Narsese 编码刺激、位置和关系上下文（如‘SAME’和‘OPPOSITE’），模拟关系学习过程。
    *   通过‘获取关系’机制，将具体交互经验抽象为一般化的关系假设（如从特定刺激对到变量占位符的泛化），支持对新刺激和新上下文的推理。
    *   引入‘关系命名’（Relational Naming），将学习到的关系显式表示为符号形式（如 `<(X1 * Y1) --> SAME>`），增强符号清晰度和推理能力。
*   **关键特点:** 该方法不依赖预定义规则，而是通过自适应学习从经验中构建关系知识，强调上下文控制而非单纯的关联强度，接近人类关系推理的灵活性。

## Experiment

*   **有效性:** 实验分为三个阶段（预训练、关系网络训练、推导关系测试），NARS 在所有阶段均达到 100% 准确率，尤其在未显式训练的推导测试阶段（Phase 3）远超随机水平（50%），证明了其通过对称性和传递性自发推导出新关系的能力。
*   **合理性:** 实验设置覆盖了从基础训练到复杂泛化的多个阶段，并通过内部置信度指标（Confidence）验证了关系框架的内化效果，设计较为全面。
*   **局限性:** 实验使用的刺激和关系网络较为简单，缺乏对复杂多关系框架或真实世界数据的测试，可能限制结果的生态效度和实际应用性。

## Further Thoughts

NARS 的‘获取关系’机制启发我们，AGI 系统可以通过与环境的交互自适应构建符号知识，而无需大量预定义规则或标注数据，这种方法可能适用于自然语言理解或复杂决策任务；此外，‘关系命名’机制为符号推理提供了清晰表示形式，未来可探索将其扩展到多关系框架（如层次、比较）或结合多模态数据（如视觉和语言）以增强泛化能力。