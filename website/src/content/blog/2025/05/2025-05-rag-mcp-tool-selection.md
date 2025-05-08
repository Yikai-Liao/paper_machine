---
title: "RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation"
pubDatetime: 2025-05-06T08:05:35+00:00
slug: "2025-05-rag-mcp-tool-selection"
type: "arxiv"
id: "2505.03275"
score: 0.6429239927941416
author: "grok-3-latest"
authors: ["Tiantian Gan", "Qiyao Sun"]
tags: ["LLM", "Retrieval-Augmented Generation", "Tool Selection", "Prompt Efficiency", "Context Management"]
institution: ["Beijing University of Post and Communications", "Queen Mary University of London"]
description: "本文提出 RAG-MCP 框架，通过检索增强生成机制动态筛选最相关工具描述，显著缓解大型语言模型在工具选择中的提示膨胀问题，并提升选择准确性。"
---

> **Summary:** 本文提出 RAG-MCP 框架，通过检索增强生成机制动态筛选最相关工具描述，显著缓解大型语言模型在工具选择中的提示膨胀问题，并提升选择准确性。 

> **Keywords:** LLM, Retrieval-Augmented Generation, Tool Selection, Prompt Efficiency, Context Management

**Authors:** Tiantian Gan, Qiyao Sun

**Institution(s):** Beijing University of Post and Communications, Queen Mary University of London


## Problem Background

大型语言模型（LLMs）在面对日益增多的外部工具（如通过 Model Context Protocol, MCP 定义的工具）时，遭遇了提示膨胀（Prompt Bloat）和工具选择复杂性（Selection Complexity）的问题。
提示膨胀导致上下文窗口 token 数量激增，超出模型处理能力，而工具数量增加则加剧了选择难度，造成性能下降和错误率上升。
论文旨在解决这一关键问题：如何在工具数量快速增长的背景下，让 LLMs 高效、准确地选择并使用相关工具，同时减少上下文负担。

## Method

*   **核心思想:** 提出 RAG-MCP 框架，通过检索增强生成（Retrieval-Augmented Generation, RAG）机制，将工具选择从模型生成任务中解耦，避免一次性将所有工具描述注入提示，而是动态筛选最相关的工具。
*   **具体实现步骤:**
    *   **外部向量索引构建:** 将所有 MCP 工具的描述（包括功能模式、用法示例等）存储在一个外部向量索引中，作为工具知识库。
    *   **语义检索:** 当用户查询到来时，使用一个轻量级检索模型（如 Qwen）对查询进行编码，通过语义搜索从索引中提取与查询最相关的 top-k 个工具候选。
    *   **验证与筛选:** 对检索到的工具进行可选的兼容性验证，通过生成少样本示例测试工具响应，确保工具的可用性和匹配度。
    *   **工具调用:** 最终仅将最相关的单个 MCP 工具描述注入到 LLM 的提示或功能调用接口中，供模型执行任务规划和操作。
*   **技术优势:** 这种方法显著减少了提示中的 token 数量，降低了模型的认知负担，同时通过外部索引的动态更新实现了工具集的可扩展性，无需重新训练模型即可添加新工具。
*   **创新点:** 相比传统方法（一次性提示所有工具描述），RAG-MCP 更像是一个‘按需取用’的工具库，通过检索机制有效缓解了提示膨胀和决策复杂性问题。

## Experiment

*   **压力测试结果:** 通过 MCP 压力测试（工具数量从 1 到 11100 个），发现传统方法（如 Blank Conditioning）随着工具数量增加，工具选择准确性和任务成功率显著下降，而 RAG-MCP 在小到中等规模工具池（<30 个工具）中成功率超过 90%，在较大规模下虽有下降但仍优于基线。
*   **基准测试效果:** 在 MCPBench 的 Web 搜索子集上，RAG-MCP 的工具选择准确率达到 43.13%，远高于基线方法 Actual Match（18.20%）和 Blank Conditioning（13.62%）；提示 token 数量从 2133.84 减少到 1084，降幅超过 50%，验证了其在减少提示膨胀上的显著效果。
*   **实验设置合理性:** 实验设计全面，涵盖了工具数量从少到多的广泛范围，并通过准确率、提示 token 数、完成 token 数等多指标评估性能；但未充分探讨多工具协同调用的场景，且在超大规模工具池中检索精度下降，提示未来优化方向。
*   **总体评价:** RAG-MCP 的提升明显，尤其在提示效率和选择准确性上表现突出，但在极端规模下的性能仍需改进。

## Further Thoughts

RAG-MCP 的检索与生成解耦思想启发了我，是否可以将这一机制推广到其他领域，如知识密集型任务中动态检索知识片段，或多模态任务中按需加载图像/视频资源？
此外，动态上下文管理的策略是否能在多轮对话中进一步优化，例如根据对话进展丢弃不相关信息，或预测性加载工具/知识？
最后，外部索引的可扩展性设计是否能演化为‘自适应索引’，根据用户行为或任务分布自动优化检索优先级，提升系统效率？