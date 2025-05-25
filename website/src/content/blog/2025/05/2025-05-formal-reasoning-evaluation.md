---
title: "Do Large Language Models Excel in Complex Logical Reasoning with Formal Language?"
pubDatetime: 2025-05-22T17:57:23+00:00
slug: "2025-05-formal-reasoning-evaluation"
type: "arxiv"
id: "2505.16998"
score: 0.7942759348153536
author: "grok-3-latest"
authors: ["Jin Jiang", "Jianing Wang", "Yuchen Yan", "Yang Liu", "Jianhua Zhu", "Mengdi Zhang", "Xunliang Cai", "Liangcai Gao"]
tags: ["LLM", "Logical Reasoning", "Formal Language", "Trajectory Format", "Generalization"]
institution: ["Peking University", "Meituan Group", "Zhejiang University"]
description: "本文通过多维度评估框架系统分析了大型语言模型在形式化语言下的逻辑推理能力，并提出拒绝采样微调方法显著提升小模型性能。"
---

> **Summary:** 本文通过多维度评估框架系统分析了大型语言模型在形式化语言下的逻辑推理能力，并提出拒绝采样微调方法显著提升小模型性能。 

> **Keywords:** LLM, Logical Reasoning, Formal Language, Trajectory Format, Generalization

**Authors:** Jin Jiang, Jianing Wang, Yuchen Yan, Yang Liu, Jianhua Zhu, Mengdi Zhang, Xunliang Cai, Liangcai Gao

**Institution(s):** Peking University, Meituan Group, Zhejiang University


## Problem Background

大型语言模型（LLMs）在复杂逻辑推理任务中表现出色，但其在形式化语言（如 Python、Z3、CSP）环境下的能力尚未得到系统性评估。
论文旨在探究 LLMs 是否真正擅长使用形式化语言解决逻辑推理问题，并识别其在不同模型类型、任务类型和轨迹格式下的表现差异。

## Method

*   **评估框架设计：** 构建了一个多维度评估体系，从三个方面分析 LLMs 的表现：
    *   **模型种类：** 包括 Thinking 模型（如 QwQ-32B）和 Instruct 模型（如 GPT-4o），参数规模从 7B 到 72B。
    *   **任务类型：** 按照逻辑推理分类为演绎（Deductive）、归纳（Inductive）、溯因（Abductive）和混合形式（Mixed-Form），涵盖 66 个数据集。
    *   **轨迹格式：** 包括自然语言（Text）和形式化语言（PoT、Z3、CSP），通过 zero-shot 设置测试模型生成轨迹的能力，并结合外部执行引擎验证结果。
*   **泛化性分析：** 通过在不同任务类型和轨迹格式上进行训练和测试，评估模型的跨域泛化能力，采用粗粒度（按任务类型或格式分组）和细粒度（任务类型与格式组合）分析。
*   **能力增强策略：** 提出拒绝采样微调（Rejected Fine-Tuning, RFT）方法，利用 GPT-4o 作为教师模型生成高质量形式化语言数据，对小型模型（如 Qwen2.5-7B）进行微调，提升其在形式化推理任务中的表现。
*   **实现细节：** 评估采用 vLLM 框架，推理配置为贪婪解码，最大生成长度 16K tokens；训练使用 Megatron-LM 框架，学习率 1e-5，批大小 128，训练 3 个 epoch。

## Experiment

*   **评估结果：** Thinking 模型（如 QwQ-32B）在大多数任务中显著优于 Instruct 模型，尤其在归纳和混合形式推理任务上；自然语言（Text）格式通常优于形式化语言，但在结构化任务（如数学计算）中 PoT 表现更好，Z3 和 CSP 在逻辑规则任务中占优；模型在复杂任务上的形式化语言表现下降明显，小模型（如 7B 规模）在形式化语言推理中表现较差。
*   **泛化性分析：** PoT 格式在跨格式迁移中表现较好（如从 PoT 到 Text 或 Z3），而 CSP 格式迁移性较差（如 CSP 到 PoT 性能下降达 -15.8）；演绎推理任务中的 CSP 格式最易被泛化，溯因推理在不同轨迹格式间迁移效果较好。
*   **能力增强效果：** 通过 RFT 方法对 Qwen2.5-7B 模型进行形式化数据增强后，平均准确率从 34.0% 提升至 42.0%，执行率从 65.3% 提升至 76.0%，在 CSP 格式上提升尤为显著（从 20.0% 到 37.0%，提升 17.0%）；增强后的小模型在所有格式上均超越开源 Instruct 模型。
*   **实验设置合理性：** 实验覆盖多种模型、任务类型和轨迹格式，数据集包括 66 个子集，评估和泛化分析设计全面；但论文指出数据集覆盖有限，未包含最新模型，形式化语言仅限于 PoT、Z3 和 CSP，未探索其他符号系统（如 Lean、Prolog）。

## Further Thoughts

论文揭示了形式化语言与任务类型的对齐性对模型表现的影响，不同任务偏好不同轨迹格式（如结构化任务偏好 PoT，逻辑规则任务偏好 Z3），这启发我们可以在推理或训练时动态选择最适合的形式化语言以优化性能；此外，RFT 方法通过拒绝采样生成高质量数据显著提升小模型能力，未来可以探索更复杂的采样策略或多模型协作生成数据，进一步提升形式化推理能力。