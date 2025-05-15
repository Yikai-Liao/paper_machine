---
title: "LongCodeBench: Evaluating Coding LLMs at 1M Context Windows"
pubDatetime: 2025-05-12T05:38:03+00:00
slug: "2025-05-longcode-benchmark"
type: "arxiv"
id: "2505.07897"
score: 0.6600079487762216
author: "grok-3-latest"
authors: ["Stefano Rando", "Yuta Kyuragi", "Alessio Sampieri", "Luca Franco", "Luca Romani", "Fabio Galasso", "John Yang", "Tatsunori Hashimoto"]
tags: ["Long Context", "Coding Benchmark", "Code Comprehension", "Code Repair", "Evaluation Framework"]
institution: ["Panasonic AI Research", "ItalAI", "Sapienza University of Rome", "Stanford University"]
description: "本文提出 LongCodeBench，一个基于真实 GitHub 数据的长上下文基准测试框架，评估大型语言模型在代码理解和修复任务上的表现，揭示了其在百万 token 上下文下的性能局限。"
---

> **Summary:** 本文提出 LongCodeBench，一个基于真实 GitHub 数据的长上下文基准测试框架，评估大型语言模型在代码理解和修复任务上的表现，揭示了其在百万 token 上下文下的性能局限。 

> **Keywords:** Long Context, Coding Benchmark, Code Comprehension, Code Repair, Evaluation Framework

**Authors:** Stefano Rando, Yuta Kyuragi, Alessio Sampieri, Luca Franco, Luca Romani, Fabio Galasso, John Yang, Tatsunori Hashimoto

**Institution(s):** Panasonic AI Research, ItalAI, Sapienza University of Rome, Stanford University


## Problem Background

近年来，大型语言模型（LLMs）的上下文窗口迅速扩展到百万 token 级别，但现有基准测试多为合成任务，难以评估长上下文语言模型（LCLMs）在真实场景中的表现。
关键问题在于，真实世界中的软件工程任务（如代码理解和修复）需要处理大规模输入（如整个代码库），而现有基准测试缺乏真实性和经济价值，无法充分揭示模型在长上下文下的能力局限。

## Method

*   **核心框架:** 提出 LongCodeBench (LCB)，一个专门为长上下文语言模型设计的基准测试框架，包含两个主要任务，评估模型在代码理解和修复上的能力，上下文长度覆盖 32K 到 1M token。
*   **任务一 - LongCodeQA:** 测试代码理解能力，通过从真实 GitHub 问题中提取的多选题，要求模型在长上下文中找到相关信息并回答问题。数据收集经过多重过滤（如使用 GPT-4 排除无需仓库特定知识的问题），确保任务需要深入理解代码库。
*   **任务二 - LongSWE-Bench:** 测试代码修复能力，要求模型根据 GitHub 问题生成修复补丁，并通过单元测试验证补丁有效性。输入包含真实文件和随机文件，模拟真实挑战，评估模型在长上下文下的生成精度。
*   **设计原则:** 方法遵循可扩展性（Scalability，适合不同规模模型）、真实性（Realism，使用真实 GitHub 数据）和长上下文（Long-context，覆盖多种上下文长度）三大原则。
*   **数据验证:** 通过手动验证和专用执行环境（如 Docker 镜像）确保任务的可靠性和可重复性，避免数据污染和评估偏差。

## Experiment

*   **有效性:** 实验测试了多个开源和闭源模型（如 Qwen2.5、Claude 3.5 Sonnet、GPT-4o、Gemini 系列）在不同上下文长度（32K 到 1M）下的表现。结果显示大多数模型在短到中上下文（32K-128K）表现较好，但随着上下文增加（如到 1M），性能显著下降，例如 Qwen2.5 在 LongCodeQA 上从 70.2% 降到 40%，Claude 3.5 Sonnet 在 LongSWE-Bench 上从 29% 降到 3%。
*   **任务难度差异:** LongCodeQA 任务上模型表现较好，而 LongSWE-Bench 由于生成任务的二元性（代码要么正确要么失败），整体性能较低，开源模型尤其表现不佳。
*   **实验设置合理性:** 实验覆盖了多种上下文长度，数据来源于真实 GitHub 仓库，避免合成数据局限性，并分析了文件长度、主题分布等对性能的影响，设置全面且合理。
*   **局限性与洞察:** 论文未提出新模型或训练方法，而是揭示了现有 LCLMs 在长上下文下的弱点，为未来改进提供了数据支持和方向。

## Further Thoughts

长上下文评估应结合真实任务（如软件工程）而非仅依赖合成数据，这启发我们可以在其他专业领域（如法律、医疗）设计类似基准测试，探索模型表现；此外，区分理解和生成任务的设计揭示了模型在不同能力上的长上下文瓶颈，提示未来优化可针对特定任务增强；最后，上下文长度与性能的非线性关系（例如文件长度较长时性能反而提升）可能与注意力机制或架构设计有关，值得进一步研究。