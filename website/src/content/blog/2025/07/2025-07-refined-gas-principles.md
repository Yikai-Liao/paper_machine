---
title: "Refining Gelfond Rationality Principle Towards More Comprehensive Foundational Principles for Answer Set Semantics"
pubDatetime: 2025-07-02T15:47:54+00:00
slug: "2025-07-refined-gas-principles"
type: "arxiv"
id: "2507.01833"
score: 0.26884291071692046
author: "grok-3-latest"
authors: ["Yi-Dong Shen", "Thomas Eiter"]
tags: ["ASP", "Non-Monotonic Reasoning", "Answer Set Semantics", "Rationality Principle", "Well-Supportedness"]
institution: ["State Key Laboratory of Computer Science, Institute of Software, Chinese Academy of Sciences", "Institute of Logic and Computation, Vienna University of Technology"]
description: "本文通过精炼 Gelfond 理性原则，提出答案集编程（ASP）的三个通用原则，并定义理性答案集和世界观语义，为 ASP 语义设计提供了更全面的基础。"
---

> **Summary:** 本文通过精炼 Gelfond 理性原则，提出答案集编程（ASP）的三个通用原则，并定义理性答案集和世界观语义，为 ASP 语义设计提供了更全面的基础。 

> **Keywords:** ASP, Non-Monotonic Reasoning, Answer Set Semantics, Rationality Principle, Well-Supportedness

**Authors:** Yi-Dong Shen, Thomas Eiter

**Institution(s):** State Key Laboratory of Computer Science, Institute of Software, Chinese Academy of Sciences, Institute of Logic and Computation, Vienna University of Technology


## Problem Background

答案集编程（ASP）作为一种声明式问题求解范式，依赖逻辑程序的答案集（Answer Sets）或世界观（World Views）来表示解，但不同类型逻辑程序的多种答案集语义缺乏统一评估标准，可能导致直观合理的解被排除；论文旨在解决是否应将最小模型性质（MM）、约束单调性（CM）和有根据性（FN）作为强制条件，以及寻找更通用的语义原则。

## Method

* **质疑传统标准：** 通过广义战略公司问题（GSC）的实例，展示传统标准（MM、CM、FN）可能排除直观合理的答案集和世界观，论证这些标准不应是强制性的。
* **精炼 Gelfond 理性原则（GAS）：** 将 Gelfond 的理性原则细化为三个新原则：答案集关于默认否定的最小性（Minimality w.r.t. Negation by Default，确保尽量不相信未被强制相信的内容）、世界观关于认知否定的最小性（Minimality w.r.t. Epistemic Negation，最大化认知否定以最小化知识）和答案集的良好支持性（Well-Supportedness，避免循环推理）。
* **扩展良好支持性定义：** 从简单正常程序扩展到析取程序和认知程序，基于严格良序偏序构建层次映射，确保答案集和世界观无循环推理。
* **定义新语义：** 提出理性答案集语义（Rational Answer Set Semantics）和理性世界观语义（Rational World View Semantics），以最小良好支持模型为核心，体现精炼的 GAS 原则。
* **评估与复杂性分析：** 使用精炼原则作为基准评估现有 ASP 语义，并分析良好支持性及新语义的计算复杂性，验证其理论可行性。

## Experiment

* **有效性验证：** 通过 GSC 问题实例，证明传统标准（MM、CM、FN）可能排除合理答案集和世界观，而新提出的理性语义能够包容这些直观解，体现了理论上的提升。
* **实验设置：** 实验主要基于逻辑程序实例分析，覆盖简单正常程序、析取程序和认知程序，设置较为全面，但缺乏大规模数据集或实际系统实现的验证。
* **复杂性分析：** 良好支持性和新语义的计算复杂性在多项式层次内（从 P 到 Σ^p_5），表明其具有强大的表达能力，但实际计算开销可能较高。
* **局限性：** 实验以理论推导和实例为主，未提供实际应用效果或与其他语义的定量比较。

## Further Thoughts

精炼 GAS 原则中关于多层次知识最小化的思想（默认否定和认知否定的最小性）可能不仅适用于 ASP，也能启发其他非单调推理领域（如默认逻辑或信念修正）如何构建分层推理框架；此外，良好支持性通过层次映射避免循环推理的思路，或许可以为深度学习中的可解释性推理提供理论支持，构建无循环依赖的推理链。