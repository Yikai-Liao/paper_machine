---
title: "Empowering Tabular Data Preparation with Language Models: Why and How?"
pubDatetime: 2025-08-03T03:00:02+00:00
slug: "2025-08-tabular-data-preparation"
type: "arxiv"
id: "2508.01556"
score: 0.6465815828664476
author: "grok-3-latest"
authors: ["Mengshi Chen", "Yuxiang Sun", "Tengchao Li", "Jianwei Wang", "Kai Wang", "Xuemin Lin", "Ying Zhang", "Wenjie Zhang"]
tags: ["LLM", "Tabular Data", "Data Preparation", "Semantic Understanding", "Prompt Engineering"]
institution: ["Antai College of Economics and Management, Shanghai Jiao Tong University", "The University of New South Wales", "Zhejiang Gongshang University"]
description: "本文系统分析了语言模型在表格数据准备中的适用性，提出‘为何’和‘如何’的框架，梳理现有技术并指明未来方向，为提升数据准备效率和质量提供了全面参考。"
---

> **Summary:** 本文系统分析了语言模型在表格数据准备中的适用性，提出‘为何’和‘如何’的框架，梳理现有技术并指明未来方向，为提升数据准备效率和质量提供了全面参考。 

> **Keywords:** LLM, Tabular Data, Data Preparation, Semantic Understanding, Prompt Engineering

**Authors:** Mengshi Chen, Yuxiang Sun, Tengchao Li, Jianwei Wang, Kai Wang, Xuemin Lin, Ying Zhang, Wenjie Zhang

**Institution(s):** Antai College of Economics and Management, Shanghai Jiao Tong University, The University of New South Wales, Zhejiang Gongshang University


## Problem Background

表格数据准备是数据驱动任务的关键步骤，旨在将原始、异构的表格转化为干净、集成且适合分析的形式，但传统方法（如规则系统和浅层学习模型）在捕捉复杂语义关系和适应多样任务需求方面存在局限性，且耗时巨大（数据科学家可能花费80%时间），容易导致‘垃圾输入，垃圾输出’（GIGO）问题。
近年来，大型语言模型（LLMs）凭借强大的语义理解能力展现出潜力，但缺乏系统性研究来明确为何LLMs适合此任务，以及如何在不同准备阶段有效应用。

## Method

*   **核心框架:** 论文系统性地探讨了语言模型（包括大型语言模型LLMs和小型语言模型SLMs）在表格数据准备中的应用，将流程分为四个核心阶段：数据获取（Data Acquisition）、数据集成（Data Integration）、数据清洗（Data Cleaning）和数据转换（Data Transformation），并分析了每个阶段的任务需求与LM能力的匹配性。
*   **应用策略:** 提出了两种主要策略：
    *   **LM-centric策略:** 直接使用LM，通过提示工程（Prompt Engineering）设计特定指令，或通过微调（Fine-tuning）针对任务数据训练模型。例如，在数据清洗中，提示LLM生成修复规则或直接填补缺失值；在数据转换中，利用提示让LLM生成代码执行格式转换。
    *   **LM-in-the-loop策略:** 将LM与其他组件结合，作为编码器（LM-as-Encoder）生成语义表示，或作为解码器（LM-as-Decoder）生成文本输出。例如，在数据获取中，LM编码查询和表格进行相似性搜索；在数据集成中，LM作为reranker提升匹配精度。
*   **具体任务实现:** 
    *   **数据获取:** 包括表发现、连接表搜索和可合并表搜索，常用‘表示-索引-搜索’范式，利用LM编码语义信息进行高效检索（如OpenDTR、DeepJoin）。
    *   **数据集成:** 包括模式匹配和实体匹配，常用提示工程让LLM作为reranker，或通过微调SLM提升匹配精度（如ReMatch、Ditto）。
    *   **数据清洗:** 包括错误检测、数据修复和数据填补，利用LM生成规则或直接修复（如IterClean），或结合图结构进行高阶信息传递（如UnIMP）。
    *   **数据转换:** 包括格式转换和语义转换，利用提示工程生成代码或逻辑形式（如DataMorpher、Auto-Formula）。
*   **优化手段:** 提出混合SLM和LLM降低计算成本，通过检索增强生成（RAG）减少幻觉，并探索多代理框架（如AutoPrep）实现自动化。

## Experiment

*   **效果概述:** 作为综述性论文，文中未提供作者自己的实验数据，而是总结了现有研究的成果，表明LM驱动的方法在语义理解和任务适应性上显著优于传统方法，尤其在模式匹配、实体匹配和数据清洗中，提示工程和微调策略提升了准确性（如通过Accuracy、F1-score等指标评估）。
*   **评估维度:** 评估主要基于数据准备准确性（直接对比准备结果与真实数据）和下游任务性能（比较准备前后模型表现），显示LM方法在复杂任务中表现更优。
*   **局限性与合理性:** 指出LM方法存在计算成本高、可能产生幻觉（不准确输出）以及对提示设计敏感等问题；现有基准多针对单一任务，缺乏覆盖整个流程的统一基准，评估全面性受限。
*   **结论:** 总体来看，LM方法在效果上提升明显，但成本和稳定性问题仍需解决，现有研究设置较为分散，需更多系统性验证。

## Further Thoughts

论文提出跨阶段交互的潜力，启发我思考是否可以通过多任务学习设计端到端LM框架，同时优化数据准备的多个阶段；此外，混合SLM和LLM以及知识蒸馏的策略让我联想到是否可以预训练小型表格专用模型，结合LLM推理能力实现高效部署；最后，基于LM的多代理自动化框架（如AutoPrep）提示是否可以引入强化学习（RLHF）机制，让代理在动态环境中自适应调整策略，进一步减少人为干预。