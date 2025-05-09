---
title: "YABLoCo: Yet Another Benchmark for Long Context Code Generation"
pubDatetime: 2025-05-07T13:42:23+00:00
slug: "2025-05-yabloco-long-context"
type: "arxiv"
id: "2505.04406"
score: 0.6478031579780803
author: "grok-3-latest"
authors: ["Aidar Valeev", "Roman Garaev", "Vadim Lomshakov", "Irina Piontkovskaya", "Vladimir Ivanov", "Israel Adewuyi"]
tags: ["LLM", "Code Generation", "Long Context", "Benchmark", "Repository"]
institution: ["Innopolis University", "St. Petersburg Department of the Steklov Institute of Mathematics", "Huawei Noah’s Ark Lab"]
description: "YABLoCo 提出一个针对 C/C++ 语言的大型代码库代码生成基准，填补长上下文评估空白，并通过实验验证上下文对 LLMs 性能的显著影响。"
---

> **Summary:** YABLoCo 提出一个针对 C/C++ 语言的大型代码库代码生成基准，填补长上下文评估空白，并通过实验验证上下文对 LLMs 性能的显著影响。 

> **Keywords:** LLM, Code Generation, Long Context, Benchmark, Repository

**Authors:** Aidar Valeev, Roman Garaev, Vadim Lomshakov, Irina Piontkovskaya, Vladimir Ivanov, Israel Adewuyi

**Institution(s):** Innopolis University, St. Petersburg Department of the Steklov Institute of Mathematics, Huawei Noah’s Ark Lab


## Problem Background

大型语言模型（LLMs）在代码生成任务中的性能评估主要集中于小规模或中等规模上下文（几千行代码），而现实世界软件项目可能包含数百万行代码（LoC），且现有基准多针对 Python 和 Java，缺乏对 C/C++ 等语言的支持。
YABLoCo 旨在解决这一问题，通过构建一个针对大型代码库（200K 到 2M LoC）的代码生成基准，评估 LLMs 在长上下文环境下的表现，特别是处理复杂函数依赖和生成可运行代码的能力。

## Method

* **数据集构建**：从四个大型 C/C++ 代码库（llvm-project、bullet3、openssl、redis）中选取 215 个函数，代码规模从 200K 到 2M LoC，函数按上下文依赖级别分类为五类（none, stdlib, file, package, project），并包含丰富的元数据如函数体、文档字符串（docstrings）、调用图等，以模拟真实软件开发场景。
* **数据质量控制**：通过自动过滤（函数长度限制在 2-15 行、测试覆盖率、去重）和手动评估（由三位程序员评估 docstring 质量）确保数据集的高质量，剔除不合适的函数，最终保留 215 个样本。
* **评估流程设计**：开发了一个可扩展的评估管道，使用 Docker 环境为每个代码库构建独立的测试环境，确保可重复性；通过运行代码库原有测试计算 pass@k 指标，同时使用 Exact Match 和 Edit Similarity 评估代码相似性；此外，开发了基于 Streamlit 的可视化工具，用于定性分析生成的代码与原始代码的差异。
* **实验设置**：测试多个 LLMs（如 CodeLlama-13B-Instruct、DeepSeekCoder-33B-Instruct、GPT-4），对比无上下文和‘oracle’上下文（提供函数依赖的完整上下文）下的性能，采用 beam search 生成多样化代码候选（k=10），以全面评估模型能力。

## Experiment

* **有效性**：实验显示 LLMs 在 YABLoCo 基准上的表现一般，无上下文情况下 pass@10 指标为 17.29%（CodeLlama）到 30.4%（GPT-4），表明长上下文代码生成任务具有较高难度。
* **上下文影响**：加入‘oracle’上下文后，模型性能显著提升，例如 CodeLlama 的 pass@10 从 17.29% 提高到 29.38%，DeepSeekCoder 从 22.42% 提高到 36.15%，验证了上下文对代码生成的重要性。
* **实验设置合理性**：实验覆盖了多个模型（开源和闭源）、不同上下文依赖级别和多种评估指标（pass@k、Exact Match、Edit Similarity），设置较为全面；Docker 环境的使用确保了测试的可重复性，但评估时间较长（单机测试需数小时）。
* **局限性**：部分代码库（如 bullet3）可能存在数据泄露（pre-training data overlap），导致 Exact Match 较高，影响评估公平性；测试覆盖率不足和 docstring 质量不一也可能影响结果准确性。

## Further Thoughts

YABLoCo 的上下文依赖分类（none 到 project）为研究上下文对代码生成的影响提供了细粒度视角，未来可以探索动态上下文选择机制，根据任务需求自动调整上下文范围；此外，合成 docstring 的尝试启发我们利用 LLMs 增强数据集质量，特别是在专业领域代码中数据稀缺的情况下；评估管道的可扩展性也提示我们可以构建通用化的代码生成评估框架，覆盖更多语言和代码库。