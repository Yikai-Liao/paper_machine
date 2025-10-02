---
title: ""Stop replacing salt with sugar!'': Towards Intuitive Human-Agent Teaching"
pubDatetime: 2025-09-29T12:00:53+00:00
slug: "2025-09-intuitive-human-agent-teaching"
type: "arxiv"
id: "2509.24651"
score: 0.702085696074743
author: "grok-3-latest"
authors: ["Nikolaos Kondylidis", "Andrea Rafanelli", "Ilaria Tiddi", "Annette ten Teije", "Frank van Harmelen"]
tags: ["Human-Agent Interaction", "Few-Shot Learning", "Symbolic Knowledge", "Incremental Learning", "Tutoring Strategy"]
institution: ["Vrije Universiteit Amsterdam", "University of Pisa"]
description: "本文提出一种直观的人机教学架构，通过符号知识注入、增量学习方法和战略性示例选择，使AI代理能在少量示例下高效学习主观任务。"
---

> **Summary:** 本文提出一种直观的人机教学架构，通过符号知识注入、增量学习方法和战略性示例选择，使AI代理能在少量示例下高效学习主观任务。 

> **Keywords:** Human-Agent Interaction, Few-Shot Learning, Symbolic Knowledge, Incremental Learning, Tutoring Strategy

**Authors:** Nikolaos Kondylidis, Andrea Rafanelli, Ilaria Tiddi, Annette ten Teije, Frank van Harmelen

**Institution(s):** Vrije Universiteit Amsterdam, University of Pisa


## Problem Background

人类能够从少量示例中快速学习新概念，而AI系统在主观任务（如个性化或敏感数据场景）中面临数据稀缺和泛化困难的问题。
论文旨在通过直观的人机教学架构，让AI代理从人类提供的少量示例中增量学习，适应用户偏好，同时减少人类教学负担。

## Method

*   **核心架构:** 提出一个直观的人机教学架构，包含三个关键组件，针对主观任务（如成分替代）设计，旨在让AI代理从少量示例中高效学习。
*   **领域知识（Domain Knowledge）:** 注入外部知识以扩展代理的任务理解能力，加速泛化。文中测试了多种成分表示方法：
    *   1-hot编码：将每个成分视为独立实体，维度等于成分数量（6632）。
    *   1-hot & FoodOn：结合FoodOn知识图谱的符号知识，扩展表示维度至10116，通过层次分类（如‘yellow bean pod’是‘plant food product’的子类）捕捉成分间的语义关系。
    *   FlavorGraph嵌入：基于成分共现和化学信息生成300维嵌入，反映成分相似性。
    *   FoodBert嵌入：基于Recipe1M数据集微调BERT模型，生成768维上下文嵌入，捕捉成分在食谱指令中的语义。
    查询表示通过加权平均计算，源成分权重为90%，其余食谱成分权重为10%，并根据tf-idf启发的描述性权重调整。
*   **学习方法（Learning Method）:** 设计增量学习方法，使代理能从单示例中学习并在推理时为候选成分排序：
    *   Baseline（源-目标频率）：基于示例中源成分到目标成分的替换频率排序，仅考虑源成分，忽略食谱上下文。
    *   P. Networks（原型网络启发）：将每个目标成分视为一个类，计算其查询表示的平均值作为原型，通过查询与原型的相似性排序，但易受语义稀释影响。
    *   Accumulative（累积表示）：累加目标成分的查询表示（而非平均），通过内积计算查询与目标的相似性得分，优先考虑常见替代品，避免信息丢失。
*   **教学策略（Tutoring Policy）:** 优化示例提供顺序以提升学习效率：
    *   Random：随机顺序提供示例，作为基准。
    *   Balanced：平衡探索与利用，将示例按源-目标对分组，通过嵌套循环和对数缩放策略选择示例，确保常见模式与多样性兼顾。
*   **应用场景:** 以成分替代任务为例，使用Recipe1MSubs数据集模拟人类输入，代理学习根据用户示例替换食谱成分。

## Experiment

*   **有效性:** Accumulative方法结合1-hot & FoodOn表示在少量示例（100个）下表现最佳，Hit@1为3.82%，Hit@10为10.49%，优于Baseline（Hit@1为3.07%，Hit@10为4.66%）；在10k示例下，Hit@10提升至45.52%，Baseline为40.08%。
*   **教学策略影响:** 使用Balanced策略时，100个示例下性能显著提升（Hit@1达10.99%，Hit@10达20.82%），约为全数据集（49k示例）性能的一半，显示示例顺序对学习效率的重大影响。
*   **局限性与合理性:** 实验设置全面，涵盖多种表示方法、学习方法和策略组合，并通过Hit@k和MRR指标评估性能；但依赖合成数据（Recipe1MSubs），缺乏真实用户交互验证，可能影响实际应用效果。此外，P. Networks方法在更多示例下性能下降，提示语义稀释问题。

## Further Thoughts

论文中符号知识（如FoodOn）在少样本学习中优于预训练嵌入的发现令人启发，是否可以在其他主观任务（如个性化推荐或医疗决策）中设计类似机制，通过显式知识减少对大规模数据的依赖？此外，Balanced教学策略强调示例顺序的重要性，是否可以结合主动学习，让代理主动请求特定示例，进一步优化人机协作效率？