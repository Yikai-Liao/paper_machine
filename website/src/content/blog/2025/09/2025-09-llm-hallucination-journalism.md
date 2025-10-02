---
title: "Not Wrong, But Untrue: LLM Overconfidence in Document-Based Queries"
pubDatetime: 2025-09-29T20:55:43+00:00
slug: "2025-09-llm-hallucination-journalism"
type: "arxiv"
id: "2509.25498"
score: 0.7209351140543918
author: "grok-3-latest"
authors: ["Nick Hagar", "Wilma Agustianto", "Nicholas Diakopoulos"]
tags: ["LLM", "Hallucination", "Journalism", "Retrieval-Augmented Generation", "Evidence Attribution"]
institution: ["Northwestern University", "University of Minnesota"]
description: "本文通过评估三种大型语言模型工具在新闻报道任务中的表现，揭示了模型幻觉主要表现为解释性过度自信，并提出新闻特有的幻觉分类扩展和工具设计建议，强调来源归属的重要性。"
---

> **Summary:** 本文通过评估三种大型语言模型工具在新闻报道任务中的表现，揭示了模型幻觉主要表现为解释性过度自信，并提出新闻特有的幻觉分类扩展和工具设计建议，强调来源归属的重要性。 

> **Keywords:** LLM, Hallucination, Journalism, Retrieval-Augmented Generation, Evidence Attribution

**Authors:** Nick Hagar, Wilma Agustianto, Nicholas Diakopoulos

**Institution(s):** Northwestern University, University of Minnesota


## Problem Background

大型语言模型（LLMs）在新闻编辑室的工作流程中应用日益广泛，但其生成的‘幻觉’（hallucination）——即看似合理但缺乏依据的陈述——对新闻行业的核心价值观如准确性和来源归属构成了威胁。
论文旨在探究 LLMs 在基于文档查询的报道任务中幻觉的频率和类型，揭示模型与新闻证据处理方式之间的根本性不匹配，并提出改进建议。

## Method

*   **研究设计**：构建一个包含 300 个文档的语料库，涵盖 TikTok 在美国的诉讼和政策相关内容，模拟新闻记者在研究和调查报道中的文档查询场景。
*   **模型选择**：评估三种工具——ChatGPT（OpenAI）、Gemini 2.5 Pro（Google）和 NotebookLM（Google），它们代表了不同的文档处理方式，包括检索增强生成（RAG）和上下文内处理。
*   **变量控制**：通过改变提示词的 specificity（从非常宽泛到非常具体，共有 5 种查询类型）和上下文大小（提供 10、100 或 300 个文档），模拟记者在实际工作中可能遇到的不同信息范围和任务需求。
*   **评估方法**：采用 Rawte 等人提出的幻觉分类框架，从方向（orientation）、类别（category）和严重程度（degree）三个维度对模型输出的每个句子进行标注；同时通过归纳性主题分析，识别新闻任务中特有的幻觉模式，如解释性过度自信。
*   **数据分析**：统计幻觉发生率，分析幻觉类型和分布，并比较不同工具在相同任务下的表现差异。

## Experiment

*   **幻觉频率**：总体上，30% 的模型输出包含至少一个幻觉；Gemini 和 ChatGPT 的幻觉率约为 40%，而 NotebookLM 仅为 13%，表明工具选择对结果有显著影响。
*   **幻觉类型**：大多数幻觉表现为‘解释性过度自信’，如对文档目的或受众的无根据描述，以及将归属意见转化为普遍陈述，而非捏造事实或数字。
*   **有效性与优越性**：NotebookLM 的低幻觉率可能得益于其 RAG 机制和显式引文功能，相比之下，ChatGPT 和 Gemini 的高幻觉率显示单纯依赖上下文处理或基本检索机制不足以满足新闻任务需求。
*   **实验设置合理性**：实验设计较为全面，涵盖了不同工具、提示词类型和上下文规模，贴近新闻工作多样性；但样本量较小（每个工具 10-15 个响应），且仅聚焦单一主题（TikTok 诉讼），可能限制结果普适性；工具系统差异也使得直接比较模型性能存在困难。

## Further Thoughts

论文提出现有幻觉分类框架不足以捕捉新闻任务中的特定错误模式，启发我们可以在其他领域（如法律、学术研究）探索领域特有的幻觉类型，并设计定制化评估框架；此外，是否可以通过训练模型以‘证据优先’的方式生成文本，例如在生成每个陈述时自动检索并标注支持证据？这可能需要结合知识图谱或更强大的检索系统；另外，设计一种‘幻觉检测器’在模型输出后实时评估其可信度并提示用户验证，也是一个值得探索的方向。