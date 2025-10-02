---
title: "Multilingual Text-to-SQL: Benchmarking the Limits of Language Models with Collaborative Language Agents"
pubDatetime: 2025-09-29T07:50:39+00:00
slug: "2025-09-multilingual-text-to-sql"
type: "arxiv"
id: "2509.24405"
score: 0.39610968960078013
author: "grok-3-latest"
authors: ["Khanh Trinh Pham", "Thu Huong Nguyen", "Jun Jo", "Quoc Viet Hung Nguyen", "Thanh Tam Nguyen"]
tags: ["LLM", "Text-to-SQL", "Multilingual Benchmark", "Collaborative Agents", "Reasoning"]
institution: ["Griffith University, Australia"]
description: "本文提出 MultiSpider 2.0 这一企业级多语言 Text-to-SQL 基准，并通过协作语言代理（COLA）框架显著提升复杂查询生成准确率，揭示了现有模型在多语言环境下的性能差距。"
---

> **Summary:** 本文提出 MultiSpider 2.0 这一企业级多语言 Text-to-SQL 基准，并通过协作语言代理（COLA）框架显著提升复杂查询生成准确率，揭示了现有模型在多语言环境下的性能差距。 

> **Keywords:** LLM, Text-to-SQL, Multilingual Benchmark, Collaborative Agents, Reasoning

**Authors:** Khanh Trinh Pham, Thu Huong Nguyen, Jun Jo, Quoc Viet Hung Nguyen, Thanh Tam Nguyen

**Institution(s):** Griffith University, Australia


## Problem Background

Text-to-SQL 技术旨在将自然语言查询转化为可执行 SQL 语句，但现有基准数据集多为英语导向，忽略了多语言环境下的需求，尤其是在企业级复杂数据库场景中，导致模型在非英语语言上的表现差距明显，限制了全球范围内的实际应用。
论文通过构建 MultiSpider 2.0 基准数据集，揭示了大型语言模型（LLMs）在多语言复杂查询上的低准确率问题（仅 4% 执行准确率，相较于 MultiSpider 1.0 的 60%），并探索提升多语言环境下的性能。

## Method

*   **核心贡献 1 - MultiSpider 2.0 基准数据集：** 扩展 Spider 2.0 的企业级复杂性，覆盖 8 种语言（英语、德语、法语、西班牙语、葡萄牙语、日语、汉语、越南语），包含 5056 个自然语言-SQL 对，涉及 200 个企业级数据库。
    *   数据集保留了复杂的 SQL 结构（如嵌套查询、多跳连接），并通过专业翻译团队和多轮验证（包括 schema 本地化和跨语言一致性检查）增加了语言及方言多样性，模拟真实世界多语言企业环境。
*   **核心贡献 2 - 协作语言代理（Collaborative Language Agents, COLA）框架：** 提出一种多代理协作基线方法，通过模块化设计提升 Text-to-SQL 性能，无需任务特定微调。
    *   **Classifier 模块：** 将大型数据库划分为更小的相关子数据库，减少无关信息干扰，提升查询精度。
    *   **Analyzer 模块：** 将复杂用户查询分解为结构化的子问题，并通过链式推理（Chain-of-Thought）生成初始 SQL 查询。
    *   **Corrector 模块：** 执行生成的 SQL 查询，基于反馈迭代修正语法和逻辑错误，确保查询功能正确性。
    *   **设计目标：** 通过结构化协作和迭代优化，弥补单一 LLM 在多语言复杂任务中的不足，特别是在 schema 链接和多跳查询上的挑战。

## Experiment

*   **挑战性：** MultiSpider 2.0 显著增加了任务难度，即使最强的 COLA + OpenAI-o1-1217 模型，其执行准确率（EX）也从 MultiSpider 1.0 的 90%+ 下降到 12-16%，降幅超过 75 个百分点，凸显了企业级 schema、语言多样性和复杂查询结构的挑战。
*   **方法提升：** COLA 框架在所有语言和模型上均显著提升性能，例如纯推理的 OpenAI-o1-1217 在 MultiSpider 2.0 上的 EX 仅为 4-5%，结合 COLA 后提升至 13-16%，特别是 Corrector 模块贡献了最大增益（从基线 5.6% 提升至最终 15.4%）。
*   **实验设置合理性：** 实验覆盖自包含解析器、纯推理 LLM 和 COLA 框架三种设置，测试了多种模型（包括 GPT-4o 驱动的解析器和多个 LLMs），并在 MultiSpider 1.0 和 2.0 上对比，使用了 Exact Matching (EM)、Execution Accuracy (EX) 和 Pass@N 等多维度指标。
*   **不足与分析：** 尽管 COLA 提升明显，但整体准确率仍较低（最高 16%），非英语语言（特别是日语、汉语、越南语）表现持续落后，反映出语言多样性和数据稀缺性问题；此外，COLA 的多代理协作导致计算成本较高，可能不适合低延迟场景。

## Further Thoughts

COLA 的多代理协作设计（Classifier, Analyzer, Corrector）展示了将复杂任务分解并通过模块化协作解决的潜力，可推广至其他 NLP 任务如多语言问答或代码生成；MultiSpider 2.0 的构建过程为设计真实世界多语言基准提供了经验，提示未来研究应关注非英语语言和文化多样性；此外，Corrector 模块利用执行反馈迭代优化的思路，启发我们可以在生成任务中引入‘执行-反馈-优化’循环，通过环境交互提升模型鲁棒性。