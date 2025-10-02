---
title: "Agentic Specification Generator for Move Programs"
pubDatetime: 2025-09-29T09:34:31+00:00
slug: "2025-09-move-specification-generator"
type: "arxiv"
id: "2509.24515"
score: 0.7739105919919579
author: "grok-3-latest"
authors: ["Yu-Fu Fu", "Meng Xu", "Taesoo Kim"]
tags: ["LLM", "Specification Generation", "Formal Verification", "Smart Contract", "Agentic Design"]
institution: ["Georgia Institute of Technology", "University of Waterloo"]
description: "本文提出 MSG，一种基于代理式设计的自动化工具，利用 LLM 为 Move 智能合约生成形式化规范，通过模块化生成、验证反馈和上下文优化，显著提升了规范的可验证性和全面性。"
---

> **Summary:** 本文提出 MSG，一种基于代理式设计的自动化工具，利用 LLM 为 Move 智能合约生成形式化规范，通过模块化生成、验证反馈和上下文优化，显著提升了规范的可验证性和全面性。 

> **Keywords:** LLM, Specification Generation, Formal Verification, Smart Contract, Agentic Design

**Authors:** Yu-Fu Fu, Meng Xu, Taesoo Kim

**Institution(s):** Georgia Institute of Technology, University of Waterloo


## Problem Background

智能合约在区块链中管理着巨额资产，任何代码漏洞可能导致重大经济损失，因此需要高保障软件开发，而形式化规范（Formal Specification）是形式化验证的基础。然而，手动编写规范对开发者来说是一个巨大负担，尤其是在新兴语言如 Move 上，代码库和文档有限，LLM 的训练数据不足，可能导致代码理解和规范生成性能下降。此外，如何利用规范语言的特性、验证工具的反馈以及评估规范的全面性，都是亟待解决的问题。

## Method

* **核心思想**：利用大型语言模型（LLM）的代码理解和生成能力，通过代理式（Agentic）设计，为 Move 智能合约自动化生成形式化规范，同时结合验证工具反馈和上下文优化，确保规范的可验证性和全面性。
* **具体实现**：
  - **代理式模块化设计**：将规范生成任务分解为多个子任务，针对 Move 规范语言（MSL）的不同类别（如 `aborts_if` 表示中止条件、`modifies` 表示全局状态修改、`ensures` 表示后置条件、循环不变量）设计专门的 ClauseGen 代理，每个代理使用定制的系统提示（Prompt），避免一次性生成所有规范导致 LLM 过载，提高生成质量。
  - **验证器循环（Verifier-in-the-Loop）**：在每轮规范生成后，使用 Move Prover 验证工具检查规范是否通过，若失败则将诊断信息（如语法错误或反例）反馈给 LLM，指导下一轮修正，最多进行 5 轮迭代。
  - **上下文构建与优化**：通过静态分析提取目标函数及其依赖（如调用函数、数据结构），并采用最佳努力的函数内联（Best-Effort Function Inlining）技术，将部分依赖函数折叠到目标函数体内，减少 LLM 跨函数推理的负担，为其提供合适的代码上下文（分为内联版本 V1 和非内联版本 V2）。
  - **规范覆盖度评估**：引入规范覆盖度（Specification Coverage）指标，通过随机删除代码部分（如 AST 节点）生成变体，若变体仍通过验证，则表明规范不完整，相关信息反馈给 LLM 以完善规范。
  - **抽象规范生成**：当无法生成具体规范时，生成抽象规范（使用未解释函数作为占位符），为专家后续完善提供基础。
* **关键点**：方法充分利用 MSL 的组合性特性，通过模块化和反馈机制，平衡 LLM 的生成能力与验证工具的精确性，同时注重上下文工程，确保生成过程自动化且无需人工干预。

## Experiment

* **有效性**：在 Aptos 区块链的 Move 代码库（包括 move-stdlib、aptos-stdlib、aptos-framework，共 357 个函数）上，MSG 成功为 84% 的函数（300/357）生成了可验证规范，与专家编写的规范相比，匹配率达 82%，并额外生成了 57% 的不同条件，总体上比专家规范多 39%，其中 33.2% 为专家遗漏的独特条款。
* **方法提升**：代理式设计显著优于一体化（All-in-One）设计，生成成功率从 53.5% 提升至 84%；Move Prover 反馈进一步提升了质量，例如在一体化设计中，30.9% 的成功案例依赖后续轮次修正，而无反馈时仅 4.8%；函数内联对部分函数有帮助（如处理合理嵌套调用），但对复杂函数可能增加理解难度，整体提升不显著。
* **实验设置合理性**：实验覆盖了多个 Move 库，包含循环等复杂情况，针对每个函数重复运行 3 次以减少 LLM 随机性影响；使用的 LLM 模型包括 OpenAI 的 o3-mini、GPT-4o 和 GPT-4o-mini，展示了不同模型的表现差异；评估指标包括可验证性（是否通过 Move Prover）和全面性（与专家规范的匹配度和覆盖度），设置全面且合理。
* **开销与局限**：主要开销在于多轮生成和 Move Prover 验证的计算成本；对于复杂函数（如深层嵌套调用）或非线性算术，生成可能失败，转而生成抽象规范；此外，LLM 偶尔会产生幻觉（如调用不存在函数），导致部分失败。

## Further Thoughts

MSG 的代理式设计启发我们可以在其他形式化验证语言中采用模块化生成策略，通过识别领域特定的规范类别（如 Solidity 的 Certora 验证语言中的中止条件和状态更新）提升生成质量；验证工具反馈机制不仅限于规范生成，还可能用于代码修复或测试用例生成；规范覆盖度评估方法提供了一种通用的质量评估思路，可应用于其他自动化生成任务；此外，上下文工程（如静态分析和函数内联）的重要性提示我们在代码生成任务中应注重上下文优化，或许可以通过动态调整上下文范围进一步提升 LLM 性能。