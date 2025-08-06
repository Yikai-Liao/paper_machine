---
title: "MArgE: Meshing Argumentative Evidence from Multiple Large Language Models for Justifiable Claim Verification"
pubDatetime: 2025-08-04T16:40:02+00:00
slug: "2025-08-marge-argumentative-ensemble"
type: "arxiv"
id: "2508.02584"
score: 0.4366957671130659
author: "grok-3-latest"
authors: ["Ming Pok Ng", "Junqi Jiang", "Gabriel Freedman", "Antonio Rago", "Francesca Toni"]
tags: ["LLM", "Ensemble Learning", "Argumentation Framework", "Claim Verification", "Reasoning"]
institution: ["Imperial College London", "King’s College London"]
description: "MArgE 框架通过整合多个大型语言模型的证据为结构化论证树，并利用计算论证语义提升声明验证的准确性和合理性。"
---

> **Summary:** MArgE 框架通过整合多个大型语言模型的证据为结构化论证树，并利用计算论证语义提升声明验证的准确性和合理性。 

> **Keywords:** LLM, Ensemble Learning, Argumentation Framework, Claim Verification, Reasoning

**Authors:** Ming Pok Ng, Junqi Jiang, Gabriel Freedman, Antonio Rago, Francesca Toni

**Institution(s):** Imperial College London, King’s College London


## Problem Background

大型语言模型（LLMs）在声明验证等任务中表现出色，但单一模型易产生幻觉等错误，而现有基于多模型协作的方法（如辩论或投票）缺乏结构化推理，导致决策过程不透明，难以追溯和合理化。

## Method

* **核心思想**：通过计算论证（Computational Argumentation）的形式化框架，将多个 LLM 的输出整合为结构化的论证树，确保声明验证决策的透明性和合理性。
* **具体步骤**：
  * **步骤1 - 论证树生成**：每个 LLM 独立针对给定声明生成支持（Pro）和反对（Con）的论证树，树结构（如深度和分支数）可定制，反映模型的推理过程。
  * **步骤2 - 论证树合并**：将多个 LLM 的论证树整合为一个统一结构，支持两种策略：简单合并（Simple Union，直接拼接所有论证）或语义合并（Semantic Merging，基于句向量相似度合并相似论证，减少冗余）。
  * **步骤3 - 论证质量评分**：使用外部 LLM（如 GPT-4o-mini）为每个论证节点分配基础分数（Base Score），评分基于相关性、事实性和模糊性等标准，分数范围在 [0, 1]，可选择是否评分根声明。
  * **步骤4 - 论证强度更新**：应用渐进语义（如 DF-QuAD）计算每个节点的辩证强度（Dialectical Strength），通过支持和反对关系的传播更新分数，最终基于根声明的强度预测真伪。
* **关键创新**：不依赖传统的思维链（Chain-of-Thought）输出，而是通过量化双极论证框架（QBAF）结构化推理，确保决策过程可检查且逻辑一致。

## Experiment

* **有效性**：MArgE 在 TruthfulClaim 和 StrategyClaim 数据集上显著优于单一 LLM、ArgLLM 集成和多模型辩论方法，准确率最高达 83.7%，在 MedClaim 数据集上稍逊（最高 73.3%），可能是由于开源模型在医学领域知识不足。
* **对比分析**：与辩论方法相比，MArgE 在大多数配置下表现更优（准确率提升高达 8.4%），且对模型顺序不敏感，稳定性更高；与 GPT-4o-mini 直接评分相比，MArgE 在两个数据集上超过其 CoT 准确率（提升 3.5% 和 1.0%）。
* **实验设置**：数据集覆盖不同推理能力（TruthfulClaim 针对常见误解，StrategyClaim 针对多跳推理，MedClaim 针对领域知识），模型包括三个开源 LLM（3.8B-8B）和 GPT-4o-mini，基线对比全面（单一 LLM、CoT 提示、辩论等），配置多样（论证深度、合并策略、评分方式）。
* **计算成本**：MArgE-D1 配置下 token 使用量低于辩论方法（输出 token 约一半），但输入 token 较多，未来可通过并行评分优化成本。

## Further Thoughts

MArgE 利用计算论证框架提升多模型协作透明度的思路令人启发，是否可以将类似形式化结构应用于其他任务（如多模态推理或生成任务）以提高可解释性？此外，语义合并策略是否能结合知识图谱等技术进一步优化论证整合，减少冗余并增强逻辑一致性？