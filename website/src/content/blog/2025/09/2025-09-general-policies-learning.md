---
title: "Learning General Policies From Examples"
pubDatetime: 2025-09-02T19:56:08+00:00
slug: "2025-09-general-policies-learning"
type: "arxiv"
id: "2509.02794"
score: 0.5479008806692309
author: "grok-3-latest"
authors: ["Blai Bonet", "Hector Geffner"]
tags: ["Generalized Planning", "Symbolic Learning", "Policy Learning", "Feature Selection", "Termination Criterion"]
institution: ["Universitat Pompeu Fabra", "RWTH Aachen University"]
description: "本文提出一种可扩展的符号方法，通过分层策略和击中集算法，在大规模规划问题中学习通用策略，显著超越传统方法的规模限制。"
---

> **Summary:** 本文提出一种可扩展的符号方法，通过分层策略和击中集算法，在大规模规划问题中学习通用策略，显著超越传统方法的规模限制。 

> **Keywords:** Generalized Planning, Symbolic Learning, Policy Learning, Feature Selection, Termination Criterion

**Authors:** Blai Bonet, Hector Geffner

**Institution(s):** Universitat Pompeu Fabra, RWTH Aachen University


## Problem Background

广义规划（Generalized Planning）旨在从一组规划问题中学习通用策略（General Policies），以解决大规模、多实例的规划任务。
传统符号方法（Symbolic Methods）生成的策略虽然可解释且正确，但受限于训练实例和特征池规模（仅数百个状态和特征），无法扩展到更大规模问题；而深度学习方法虽具扩展性，但策略不透明且泛化能力有限。
本文致力于解决如何在符号框架下开发一种可扩展的学习方法，处理包含百万状态和数十万特征的大规模训练实例和特征池，同时确保策略的正确性和泛化能力。

## Method

*   **核心思想:** 通过采样计划的泛化（Generalization of Sampled Plans）学习通用策略，并引入结构化终止条件（Structural Termination Criterion）确保策略无循环（Acyclicity），同时提升方法的可扩展性。
*   **具体实现:**
    *   **计划生成与泛化:** 使用经典规划器（Planner）生成初始计划（Plans），并基于这些计划进行泛化，形成适用于多实例的策略。
    *   **分层策略（Stratified Policies）:** 提出一种新型策略结构，通过特征的单调性（Monotonicity）和条件单调性（Conditional Monotonicity）分层排列，确保策略在设计上具有终止性，避免无限循环。
    *   **击中集算法（GENEX）:** 采用基于最小成本击中集（Min-Cost Hitting Set）的启发式算法，替代传统 SAT/ASP 求解器，高效处理大规模特征池和状态空间，生成满足终止性条件的策略。
    *   **包装算法（WRAPPER）:** 设计一个外部迭代算法，通过动态调整‘好的’（X[+]）和‘坏的’（X[-]）状态转移集，调用 GENEX 确保策略的闭合性（Closedness）和安全性（Safeness），最终生成解决整个训练集的策略。
*   **关键创新:** 将策略学习分解为基本学习任务（BLT）和元学习任务（MLT），通过高效算法实现规模扩展，同时内置终止性保证，避免传统方法中全局无循环约束的计算复杂性。

## Experiment

*   **有效性:** 在 34 个标准规划领域（Domains）中，20 个领域成功学习到通用策略，其中 14 个仅需单个计划即可泛化，表明方法在策略学习上的高效性。
*   **规模提升:** 实验处理了特征池规模高达数十万、状态空间达百万级别的实例，远超传统符号方法（通常仅处理数百特征和状态），解决了之前方法无法处理的领域。
*   **时间开销:** 大多数领域在几分钟内完成学习（如 Blocks4ops-clear 仅需 10.4 秒），少数复杂领域（如 Sokoban-1stone-7x7）耗时较长（约 7 小时），但仍展示出可行性。
*   **泛化能力:** 在更大规模测试实例上表现良好，例如 Blocks4ops 策略在训练集（10 个块）外扩展到 45 个块的实例仍保持 100% 覆盖率，表明策略的鲁棒性。
*   **不足与局限:** 在 14 个领域未能找到策略，主要因特征池表达能力不足（11 个领域）或超时（3 个领域），提示未来需改进特征生成机制。
*   **实验设置合理性:** 实验涵盖多种挑战类型领域，通过有效宽度（Effective Width）等指标评估策略质量，数据支持方法在可扩展性和泛化性上的显著优势。

## Further Thoughts

分层策略（Stratified Policies）的概念非常具有启发性，其通过特征依赖关系构建终止性（Termination）的思想可扩展至强化学习中的奖励设计或策略优化，确保复杂环境下的策略稳定性；此外，击中集算法（Hitting Set Algorithm）在处理大规模特征池时的效率，启发我们可以在其他符号学习或特征选择任务中应用类似贪心策略，平衡计算复杂度和解的质量；最后，特征池表达能力不足的问题提示是否可结合深度学习（如图神经网络）生成更丰富特征，以提升符号方法的适用范围。