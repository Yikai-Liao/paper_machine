---
title: "RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation"
pubDatetime: 2025-05-06T08:05:35+00:00
slug: "2025-05-rag-mcp-tool-selection"
type: "arxiv"
id: "2505.03275"
score: 0.6429239927941416
author: "grok-3-latest"
authors: ["Tiantian Gan", "Qiyao Sun"]
tags: ["LLM", "Tool Selection", "Retrieval-Augmented Generation", "Context Management", "Scalability"]
institution: ["Beijing University of Post and Communications", "Queen Mary University of London"]
description: "本文提出 RAG-MCP 框架，通过检索增强生成技术动态筛选相关工具描述，显著缓解了大型语言模型在大规模工具使用中的提示膨胀问题，并大幅提升了工具选择准确率。"
---

> **Summary:** 本文提出 RAG-MCP 框架，通过检索增强生成技术动态筛选相关工具描述，显著缓解了大型语言模型在大规模工具使用中的提示膨胀问题，并大幅提升了工具选择准确率。 

> **Keywords:** LLM, Tool Selection, Retrieval-Augmented Generation, Context Management, Scalability

**Authors:** Tiantian Gan, Qiyao Sun

**Institution(s):** Beijing University of Post and Communications, Queen Mary University of London


## Problem Background

大型语言模型（LLMs）在工具使用场景中面临提示膨胀（Prompt Bloat）问题：随着外部工具数量（如通过 Model Context Protocol, MCP 定义的工具）增加，将所有工具描述纳入提示会占用大量上下文窗口资源，并增加决策复杂性，导致工具选择准确率下降和潜在的幻觉或错误调用问题。
这一挑战在工具数量扩展到数十甚至上千时尤为突出，亟需一种方法来减轻提示负担并提升选择效率。

## Method

*   **核心思想:** 提出 RAG-MCP 框架，通过检索增强生成（Retrieval-Augmented Generation, RAG）技术动态筛选与用户查询最相关的工具描述，避免将所有工具一次性纳入提示，从而缓解提示膨胀和决策复杂性。
*   **具体实现步骤:** 
    *   **外部向量索引构建:** 将所有 MCP 工具的描述（包括功能模式、用法示例等）存储在一个外部向量索引中，作为工具知识库。
    *   **语义检索:** 当用户查询到来时，使用一个轻量级语言模型（如 Qwen）对查询进行编码，通过语义搜索从索引中检索出 top-k 个最相关的工具候选。
    *   **验证与筛选:** 对检索出的工具进行可选的兼容性验证（通过生成少样本示例测试工具响应），最终选择最相关的单个工具描述注入到 LLM 的提示或功能调用接口中。
    *   **任务执行:** LLM 基于筛选后的工具描述进行任务规划和执行，避免了工具发现的负担。
*   **技术优势:** 该方法显著减少了提示大小，降低了模型的认知负担，并通过外部索引的更新实现了工具集的扩展性，无需重新训练模型。
*   **创新点:** 将 RAG 技术从传统的知识检索扩展到工具选择场景，实现了工具发现与文本生成的解耦。

## Experiment

*   **压力测试（Stress Test）:** 设计了一个类似‘针在草堆中’的 MCP 压力测试，评估工具数量从 1 到 11100 变化时 LLM 的选择准确率。结果表明，传统方法（将所有工具描述放入提示）性能随工具数量增加急剧下降，而 RAG-MCP 有效缓解了这一问题，尤其在中小规模工具池中表现优异。
*   **基准测试（Benchmark Test）:** 在 MCPBench 的 Web 搜索子集上，RAG-MCP 的工具选择准确率达到 43.13%，远高于基线方法 Actual Match（18.20%）和 Blank Conditioning（13.62%）。同时，平均提示 token 数从 2133.84 降低到 1084.00，减少超过 50%。尽管完成 token 数略增（78.14 vs 23.60），但与更高的准确率和任务成功率相关，属于合理权衡。
*   **实验设置评价:** 实验设计较为全面，涵盖了工具数量从小到大的广泛范围，并通过准确率、提示 token 数和完成 token 数等多指标评估性能。压力测试和基准测试结合验证了方法的有效性，但在大规模工具池（>1000）中检索精度下降的问题表明仍有改进空间。

## Further Thoughts

RAG-MCP 的检索与生成解耦思想非常具有启发性，不仅适用于工具选择，还可以扩展到其他需要大规模上下文管理的场景，如知识密集型任务或多轮对话中的上下文压缩。
此外，外部索引的扩展性设计为构建可扩展 AI 代理提供了新思路，未来可以探索如何将类似机制应用于动态知识库或 API 集的管理。
最后，语义检索在非传统场景中的应用潜力值得关注，可以进一步研究分层索引或自适应检索策略，以应对超大规模工具集的挑战。