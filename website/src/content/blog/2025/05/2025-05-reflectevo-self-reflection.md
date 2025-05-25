---
title: "ReflectEvo: Improving Meta Introspection of Small LLMs by Learning Self-Reflection"
pubDatetime: 2025-05-22T10:03:05+00:00
slug: "2025-05-reflectevo-self-reflection"
type: "arxiv"
id: "2505.16475"
score: 0.7169367086985202
author: "grok-3-latest"
authors: ["Jiaqi Li", "Xinyi Dong", "Yang Liu", "Zhizhuo Yang", "Quansen Wang", "Xiaobo Wang", "Songchun Zhu", "Zixia Jia", "Zilong Zheng"]
tags: ["LLM", "Self-Reflection", "Reasoning", "Self-Training", "Meta Introspection"]
institution: ["State Key Laboratory of General Artificial Intelligence, BIGAI", "Peking University"]
description: "本文提出ReflectEvo框架，通过自动生成大规模自我反思数据并结合自我训练，显著提升小型语言模型的推理能力，为低成本改进SLMs开辟了新路径。"
---

> **Summary:** 本文提出ReflectEvo框架，通过自动生成大规模自我反思数据并结合自我训练，显著提升小型语言模型的推理能力，为低成本改进SLMs开辟了新路径。 

> **Keywords:** LLM, Self-Reflection, Reasoning, Self-Training, Meta Introspection

**Authors:** Jiaqi Li, Xinyi Dong, Yang Liu, Zhizhuo Yang, Quansen Wang, Xiaobo Wang, Songchun Zhu, Zixia Jia, Zilong Zheng

**Institution(s):** State Key Laboratory of General Artificial Intelligence, BIGAI, Peking University


## Problem Background

小型语言模型（SLMs）由于缺乏高质量反思数据和高成本的人工标注，难以通过自我反思提升元内省（Meta Introspection）和推理能力。
本文旨在探索SLMs是否能通过自我生成的反思数据有效学习反思能力，并解决如何利用高低质量自我生成数据进行自我训练以增强推理能力的问题。

## Method

*   **核心思想:** 提出ReflectEvo管道，通过自动生成自我反思数据并结合自我训练，增强SLMs的元内省和推理能力，而不依赖外部强模型或人工标注。
*   **反思生成:** 设计生成器（Generator）和反思器（Reflector）框架，生成器基于基础SLM生成初始答案和推理过程，反思器对错误答案进行反思并提出改进策略，反思过程包括验证失败方案、定位错误原因和制定纠正策略三个阶段。
*   **数据构建:** 构建大规模数据集ReflectEvo-460k，包含46万条自我生成的反思样本，覆盖多领域任务，通过过滤高质量反思（正确修正的答案）和利用GPT-4o进行偏好标注，形成多个训练子集。
*   **反思学习:** 采用监督微调（SFT）和直接偏好优化（DPO）进行训练，设置包括单阶段训练（同时优化反思和纠正）、双阶段训练（分别优化反思和纠正）以及利用正负样本的偏好学习，旨在提升模型对高质量反思的识别和错误纠正能力。
*   **推理过程:** 在推理时，模型作为反思器进行多轮反思和纠正，直到答案正确或达到最大轮数限制。

## Experiment

*   **有效性:** ReflectEvo显著提升了SLMs的推理能力，例如Llama-3-8B在BIG-bench上的准确率从52.4%提升至71.2%，Mistral-7B从44.4%提升至71.1%，在某些任务上甚至超越更大规模模型。
*   **对比分析:** 相比基于提示的反思和直接SFT，ReflectEvo的自我训练方法在多轮反思后展现出更大提升（∆(t1,t2)高达20%以上），尤其在逻辑推理和常识任务上效果显著。
*   **实验设置合理性:** 实验覆盖多模型（Llama-3, Mistral, Gemma-2）、多任务（LogiQA, MATH, MBPP, BIG-bench）和多训练设置（单阶段、双阶段、DPO），数据来源多样（17个子数据集），并测试了跨任务和跨模型泛化性，设置全面合理。
*   **局限性:** 在MATH和MBPP等需要细粒度推理的任务上提升有限，可能是由于反思数据缺乏针对性；自我生成数据的质量依赖于SLM初始能力，可能导致部分低质量反思影响训练效果。

## Further Thoughts

ReflectEvo的自我训练框架启发我们思考是否可以将迭代反思机制扩展到多模态任务（如视觉-语言模型），或通过动态调整反思深度和引入领域知识定制反思指令，进一步提升特定任务效果；此外，反思质量与推理改进的正相关性提示未来可以设计更精细的反馈机制，如优化外部验证器或结合奖励函数增强反思学习。