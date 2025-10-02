---
title: "DeepJSONEval: Benchmarking Complex Nested JSON Data Mining for Large Language Models"
pubDatetime: 2025-09-30T08:18:20+00:00
slug: "2025-09-deep-json-eval"
type: "arxiv"
id: "2509.25922"
score: 0.7186248811295007
author: "grok-3-latest"
authors: ["Zhicheng Zhou", "Jing Li", "Suming Qiu", "Junjie Huang", "Linyuan Qiu", "Zhijie Sun"]
tags: ["LLM", "Data Mining", "Structured Output", "Benchmarking", "Nested JSON"]
institution: ["Huawei Technologies Co., Ltd"]
description: "本文提出DeepJSONEval，一个针对深层嵌套JSON结构的评估基准，通过创新的模式生成算法和多维度评估框架，系统性地揭示了大型语言模型在复杂数据提取任务中的能力与局限。"
---

> **Summary:** 本文提出DeepJSONEval，一个针对深层嵌套JSON结构的评估基准，通过创新的模式生成算法和多维度评估框架，系统性地揭示了大型语言模型在复杂数据提取任务中的能力与局限。 

> **Keywords:** LLM, Data Mining, Structured Output, Benchmarking, Nested JSON

**Authors:** Zhicheng Zhou, Jing Li, Suming Qiu, Junjie Huang, Linyuan Qiu, Zhijie Sun

**Institution(s):** Huawei Technologies Co., Ltd


## Problem Background

互联网上充斥着低密度、高冗余的信息（如社交媒体评论、重复新闻），使得高效提取有价值信息变得困难。
多层嵌套JSON结构通过语义丰富的层次表示提供了一种解决方案，但现有大型语言模型（LLMs）的评估基准更多关注JSON生成而非数据理解与提取能力，缺乏与实际网络数据挖掘任务的相关性。
因此，亟需一个系统性评估框架来衡量LLMs在复杂嵌套JSON结构中的信息提取能力，以适应真实世界的应用需求。

## Method

*   **核心思想:** 提出DeepJSONEval，一个专注于多语言、深层嵌套JSON结构的评估基准，旨在全面评估LLMs从非结构化文本中提取信息并生成复杂JSON结构的能力。
*   **具体实现流程:** 
    *   **网络文本收集与多文档聚合:** 通过LLMs对来自异构来源的网络文本进行多文档聚合与重写，消除冗余并生成信息密集的摘要，确保输入文本的质量和多样性（至少1500字）。
    *   **模式树构建:** 从文本中提取关键概念，构建层次化的模式树，节点代表属性，边表示父子关系，确保语义关系的准确表示。
    *   **实时路径值更新束搜索算法:** 提出一种创新的算法（Real-time Path-Value Updating Beam Search），通过迭代扩展路径并实时更新路径值，从模式树中提取满足深度和属性数量约束的子树，用于构建复杂嵌套JSON模式（3-7层深度，平均17.5个属性）。该算法结合关联分数、边际贡献和奖励机制，确保子树的高质量和结构有效性。
    *   **模式生成与基准真值构建:** 将子树转换为正式JSON模式，按嵌套深度分类为中等（3-4层）和困难（5-7层）难度，并通过领域专家和人工循环验证生成可靠的基准真值（Gold标准）。
*   **评估框架:** 设计多维度评估指标，包括语法分数（Syntax Score，评估JSON语法有效性）、层次键匹配分数（Hierarchical Key Matching Score，基于Jaccard相似度评估属性匹配度）和严格分数（Strict Score，评估输出与真值的完全一致性），从多个角度分析模型性能。
*   **创新点:** 强调深层嵌套结构、全面数据类型覆盖（字符串、数字、布尔值、枚举、列表等）以及多领域应用，确保评估的复杂性和现实适用性。

## Experiment

*   **有效性:** 在DeepJSONEval的2100个多领域实例上测试了12个领先LLMs（如Claude Sonnet 4, DeepSeek R1, Gemini 2.5 Pro），结果显示所有模型在处理深层嵌套结构（5-7层）时性能显著下降，严格评估分数低于60%，表明基准对模型能力的区分度较高。
*   **全面性与合理性:** 实验设置覆盖多难度（中等和困难）、多领域（10个领域，如旅游推广、电子设备介绍）和多数据类型（字符串、数字、列表等），并通过响应长度分析证明评估结果与输出冗长度无关，体现了框架的公平性。
*   **跨领域一致性:** 模型在10个领域的表现一致（中等难度分数0.776-0.867，困难难度分数0.474-0.540），验证了基准的生态有效性，适用于不同应用场景。
*   **数据类型挑战:** 模型在复杂嵌套列表结构上的性能显著低于基本数据类型（如数字列表准确率0.90-1.00，字符串列表仅0.576-0.722），揭示了当前LLMs在处理层次结构时的局限性。
*   **外部验证:** 通过端到端网络数据提取管道实验，DeepJSONEval分数与实际任务表现高度相关（相关系数0.987），证明其在真实应用中的预测能力。
*   **局限性:** 样本量（2100个实例）虽然较大，但某些领域或难度级别的分布可能不足，未来可进一步扩展。

## Further Thoughts

DeepJSONEval的设计理念为评估复杂任务提供了新思路：首先，针对特定任务（如JSON提取）的基准设计可以显著提升评估的针对性和实用性，未来可应用于其他结构化数据任务，如知识图谱构建；其次，难度分级和多维度评估框架为评估模型在不同复杂性下的表现提供了参考，或许可以扩展到评估模型的推理深度或跨领域适应性；最后，实时路径值更新束搜索算法不仅适用于JSON模式生成，还可能用于其他层次化数据表示任务，如复杂数据库模式设计或语义网络构建。