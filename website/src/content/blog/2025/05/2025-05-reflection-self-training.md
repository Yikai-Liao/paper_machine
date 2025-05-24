---
title: "ReflectEvo: Improving Meta Introspection of Small LLMs by Learning Self-Reflection"
pubDatetime: 2025-05-22T10:03:05+00:00
slug: "2025-05-reflection-self-training"
type: "arxiv"
id: "2505.16475"
score: 0.7169367086985202
author: "grok-3-latest"
authors: ["Jiaqi Li", "Xinyi Dong", "Yang Liu", "Zhizhuo Yang", "Quansen Wang", "Xiaobo Wang", "Songchun Zhu", "Zixia Jia", "Zilong Zheng"]
tags: ["LLM", "Self-Training", "Reflection", "Reasoning", "Meta Introspection"]
institution: ["State Key Laboratory of General Artificial Intelligence (BIGAI)", "Peking University"]
description: "ReflectEvo 通过自生成反思数据和自训练显著提升了小型语言模型的元内省与推理能力。"
---

> **Summary:** ReflectEvo 通过自生成反思数据和自训练显著提升了小型语言模型的元内省与推理能力。 

> **Keywords:** LLM, Self-Training, Reflection, Reasoning, Meta Introspection

**Authors:** Jiaqi Li, Xinyi Dong, Yang Liu, Zhizhuo Yang, Quansen Wang, Xiaobo Wang, Songchun Zhu, Zixia Jia, Zilong Zheng

**Institution(s):** State Key Laboratory of General Artificial Intelligence (BIGAI), Peking University


## Problem Background

小型语言模型（SLMs）在元内省（meta introspection）和推理能力上相较大型语言模型（LLMs）存在显著差距，传统提升方法依赖于大型模型蒸馏或高成本的人工标注数据，难以扩展；
本文旨在探索是否可以通过自生成反思数据和自训练有效提升 SLMs 的自我反思能力，从而增强其推理性能，减少对外部资源的依赖。

## Method

*   **核心思想:** 提出 ReflectEvo 管道，通过自生成反思数据和自训练提升 SLMs 的元内省能力，使其能够通过迭代反思改进推理和错误纠正。
*   **反思生成:** 设计生成器（Generator）和反思器（Reflector）框架，生成器基于基础 SLM 提供初始答案和推理过程，反思器对错误答案进行反思，定位问题并提出改进策略；反思指令池包含三个关键阶段（验证失败方案、定位错误原因、制定纠正策略），确保反思的全面性和多样性。
*   **数据构建与筛选:** 构建大规模数据集 ReflectEvo-460k，包含 46 万个反思样本，覆盖 17 个源数据集和 10 个任务领域；通过过滤高质量反思（成功纠正的答案）、利用 GPT-4o 选择偏好数据以及整合正负样本，优化训练数据质量。
*   **反思学习:** 采用监督微调（SFT）和直接偏好优化（DPO）进行自训练；SFT 分为单阶段（同时训练反思和纠正）和双阶段（分别训练反思与纠正）设置，DPO 利用正负样本对进行偏好学习，提升反思质量；训练过程不依赖外部强大模型，注重 SLM 自身的迭代改进。
*   **推理阶段:** 在推理时，模型作为反思器进行多轮反思和纠正，直至答案正确或达到最大轮数限制。

## Experiment

*   **性能提升:** ReflectEvo 显著提升了 SLMs 的推理能力，例如 Llama-3 准确率从 52.4% 提升至 71.2%，Mistral 从 44.4% 提升至 71.1%，在 BIG-bench 上甚至超越更大规模模型；∆(t1,t2)（第一轮到第二轮准确率提升）在多个任务上超过 20%，显示反思学习对错误纠正的高效性。
*   **实验设置合理性:** 实验覆盖多个 SLMs（如 Llama-3-8B, Mistral-7B, Gemma-2-9B）和任务领域（LogiQA, MATH, MBPP, BIG-bench），对比了多种训练设置（单阶段 SFT、双阶段 SFT、DPO）及基线方法（如 STaR, Re-ReST），验证了方法的泛化性和鲁棒性；多轮反思实验显示性能持续改进，尤其在逻辑推理任务上表现突出。
*   **局限性与差异:** 在 MATH 和 MBPP 等任务上提升幅度较小，可能因这些任务需要更细粒度的步骤批判，而自生成反思质量受限于 SLMs 初始能力；实验还探讨了不同验证器和反思来源的影响，表明方法效果因任务类型和模型能力而异。

## Further Thoughts

自训练与迭代改进的思路可推广至其他 AI 领域，如强化学习或多智能体系统，探索无监督环境下的自我进化潜力；反思数据的多样性和质量控制启发更精细的提示工程或自动化数据筛选机制；任务特异性反思需求的差异提示未来可设计自适应反思策略或结合领域知识增强反思深度。