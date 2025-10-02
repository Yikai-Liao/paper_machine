---
title: "SING-SQL: A Synthetic Data Generation Framework for In-Domain Text-to-SQL Translation"
pubDatetime: 2025-09-30T02:14:49+00:00
slug: "2025-09-sing-sql-synthetic-data"
type: "arxiv"
id: "2509.25672"
score: 0.5586868585315555
author: "grok-3-latest"
authors: ["Hasan Alp Caferoğlu", "Mehmet Serhat Çelik", "Özgür Ulusoy"]
tags: ["LLM", "Text-to-SQL", "Synthetic Data", "Schema Linking", "In-Domain Training"]
institution: ["Bilkent University"]
description: "SING-SQL 提出了一种两阶段合成数据生成框架，为企业级Text-to-SQL任务提供高质量领域内数据，并通过 SingSQL-LM 模型验证了显著的性能提升。"
---

> **Summary:** SING-SQL 提出了一种两阶段合成数据生成框架，为企业级Text-to-SQL任务提供高质量领域内数据，并通过 SingSQL-LM 模型验证了显著的性能提升。 

> **Keywords:** LLM, Text-to-SQL, Synthetic Data, Schema Linking, In-Domain Training

**Authors:** Hasan Alp Caferoğlu, Mehmet Serhat Çelik, Özgür Ulusoy

**Institution(s):** Bilkent University


## Problem Background

Text-to-SQL 技术旨在帮助非技术用户通过自然语言查询数据库，但现有研究多聚焦于跨域泛化，忽略了企业场景中对单一数据库的高精度需求。
企业需要针对特定数据库schema定制模型并进行评估，而缺乏高质量的领域内（in-domain）数据成为关键障碍，SING-SQL 框架旨在解决这一问题：为任意目标数据库生成高质量、高覆盖率的合成Text-to-SQL数据，无需依赖SQL日志或人工标注。

## Method

*   **核心思想:** 提出一个两阶段的合成数据生成框架，针对特定数据库生成高质量、高覆盖率的Text-to-SQL数据，并通过微调紧凑语言模型提升领域内性能。
*   **第一阶段 - 子schema生成:** 
    *   将数据库schema分层分解为子schema（sub-schemas），包括表级（table-level）和列级（column-level）分解。
    *   表级分解基于外键约束提取可连接的表组合，限制每个子schema的表数量以确保实际相关性和计算可行性。
    *   列级分解通过滑动窗口策略（window size 和 stride 参数控制）选择非关键列，同时保留关键列（如主键和外键）以确保关系完整性，最终通过笛卡尔积生成多样化的子schema。
    *   这种分层分解确保了生成的子schema既可控又多样，同时覆盖整个数据库schema，降低了生成噪声。
*   **第二阶段 - 合成Text-to-SQL数据生成:** 
    *   针对每个子schema，使用大型语言模型（LLM）生成不同复杂度的SQL查询（简单、中等、挑战、窗口函数），并将其翻译为自然语言问题。
    *   质量控制流程包括：
        - LLM-as-a-judge验证逻辑一致性和语义对齐，剔除不合格的SQL-Text对。
        - 检查SQL查询的可执行性，对不可执行查询进行自动修复，若仍不可执行则剔除。
        - 为通过验证的SQL-Text对生成推理轨迹（reasoning traces），采用分治提示策略以增强可解释性和教学价值。
        - 通过列频次分析进行列平衡生成，针对低频列所在的子schema进行额外数据生成，确保schema覆盖的均匀性。
*   **模型微调:** 基于生成的合成数据，使用参数高效的LoRA方法对紧凑语言模型（如 Qwen2.5-Coder 1.5B 和 3B）进行监督微调，生成 SingSQL-LM 模型系列，适应企业级低资源环境。

## Experiment

*   **有效性:** SingSQL-LM 在 BIRD 基准测试（California Schools 数据集）上显著优于同规模基线模型。例如，SingSQL-LM-3B-R64 在 32 个候选查询下达到 82.87% Soft F1 和 73.03% Execution Accuracy（EX），分别比最佳 3B 基线高出 16.21% 和 12.36%；在 1.5B 规模下，SingSQL-LM-1.5B-R64 比基线提升了 9.30% Soft F1 和 4.49% EX。在自生成合成数据集上，性能优势更加明显，表明领域内数据对模型性能的提升至关重要。
*   **实验设置合理性:** 实验覆盖了不同模型规模（1.5B 和 3B）、不同候选查询数量（8、16、32），并对比了多种基线模型（如 CODES、SLM-SQL、CscSQL 等）。此外，测试了schema过滤性能（列召回率高达 97.91%）和上下文管理策略（如 schema-free 微调结合 schema-only 推理），设置全面且贴近实际应用场景。
*   **局限性与开销:** 实验未涉及强化学习等复杂后训练方法，也未探讨解码参数的影响，可能限制进一步性能提升；合成数据生成和微调过程依赖于 LLM（如 Gemini-2.5-Flash），增加了计算成本，但通过 LoRA 微调降低了资源需求。

## Further Thoughts

子schema分解策略非常有启发性，通过分层分解复杂数据库为可控子schema，不仅降低了生成噪声，还确保了全面覆盖，这种思想可推广至其他结构化数据任务如知识图谱查询；此外，列平衡生成针对低频列额外生成数据以确保分布均匀，这一方法可在数据稀疏场景（如长尾分类）中应用；最后，schema-free 微调结合 schema-only 推理的优越性能提示上下文依赖设计的重要性，可能对代码生成等任务有借鉴价值。