---
title: "Role-Aware Language Models for Secure and Contextualized Access Control in Organizations"
pubDatetime: 2025-07-31T11:41:04+00:00
slug: "2025-07-role-aware-access-control"
type: "arxiv"
id: "2507.23465"
score: 0.4710346293072492
author: "grok-3-latest"
authors: ["Saeed Almheiri", "Yerulan Kongrat", "Adrian Santosh", "Ruslan Tasmukhanov", "Josemaria Vera", "Muhammad Dehan Al Kautsar", "Fajri Koto"]
tags: ["LLM", "Access Control", "Role-Based Security", "Fine-Tuning", "Enterprise Application"]
institution: ["Mohamed bin Zayed University of Artificial Intelligence", "Nazarbayev University", "University of Illinois at Urbana-Champaign", "New York University Abu Dhabi"]
description: "本文提出并评估了角色感知语言模型，通过三种建模策略（BERT 分类器、LLM 分类器和生成模型）实现企业环境中基于角色的访问控制，显著提升了 LLMs 的安全性和上下文适应性。"
---

> **Summary:** 本文提出并评估了角色感知语言模型，通过三种建模策略（BERT 分类器、LLM 分类器和生成模型）实现企业环境中基于角色的访问控制，显著提升了 LLMs 的安全性和上下文适应性。 

> **Keywords:** LLM, Access Control, Role-Based Security, Fine-Tuning, Enterprise Application

**Authors:** Saeed Almheiri, Yerulan Kongrat, Adrian Santosh, Ruslan Tasmukhanov, Josemaria Vera, Muhammad Dehan Al Kautsar, Fajri Koto

**Institution(s):** Mohamed bin Zayed University of Artificial Intelligence, Nazarbayev University, University of Illinois at Urbana-Champaign, New York University Abu Dhabi


## Problem Background

随着大型语言模型（LLMs）在企业环境中的广泛应用，基于用户角色的访问控制成为一个迫切需求。
传统安全方法通常假设用户访问权限统一，专注于防止有害输出，而忽略了组织中角色特定的访问限制，可能导致敏感信息泄露。
论文旨在探索如何通过微调 LLMs，使其生成符合组织角色权限的响应，从而增强安全性和上下文适应性。

## Method

*   **核心思想:** 开发角色感知的语言模型，使其根据用户角色执行访问控制，限制未经授权的信息披露，同时保持响应质量。
*   **具体策略:** 提出了三种建模方法：
    *   **BERT-based Classifier (Role-aware Cls):** 使用 BERT 系列模型（如 Modern BERT, Google BERT, RoBERTa 的 Base 和 Large 版本）进行二元分类，判断特定角色是否被授权访问某个指令。角色信息以 '<prompt> [SEP] <role>' 形式附加到提示后，模型输出访问许可与否。
    *   **LLM-based Classifier (Role-aware LLM-Cls):** 基于开源大型语言模型（如 Qwen 2.5 3B/7B, Llama 3.x 3B/8B, Gemma 4B/7B）进行微调，同样进行二元分类任务。角色信息以 'Position: <role> <prompt>' 形式前置于提示，结合系统指令，模型通过 LoRA 适配器进行监督学习，输出 True（授权）或 False（拒绝）。
    *   **Role-aware LLM-Gen:** 使用相同的 LLMs 和微调设置，但训练模型直接生成完整响应而非分类结果。角色信息同样前置，但无系统提示，模型根据角色权限生成适当回答或拒绝信息，输出更贴近自然语言。
*   **数据集构建:** 设计了两种互补数据集：
    *   **改编数据集:** 从现有指令调优数据集（如 Databricks Dolly-15k）通过聚类和角色标注改编，基于组织层次结构分配指令权限。
    *   **合成数据集:** 使用 GPT-4.1 mini 生成角色敏感的企业场景数据，模拟真实组织交互，包含两种组织结构（Basic 和 Office）以测试不同层次复杂性。
*   **角色编码:** 探索了三种编码方式（Hierarchical Number Encoding, Single Name Encoding, Hierarchical Name Encoding）以表示角色层次，分析其对访问控制的影响。

## Experiment

*   **有效性:** Role-aware LLM-Cls 方法在访问控制准确性上表现最佳，在改编数据集（Dolly）上 Modern BERT Large 达到 90.0% 准确率，在合成数据集上 Llama 3.1 8B Instruct 达到 89.3%；Role-aware LLM-Gen 准确率略低（约 5-10 个百分点），但生成质量评分高（正确性、完整性、清晰度均接近 4/5）。
*   **全面性与合理性:** 实验覆盖多种模型（不同规模的 BERT 和 LLMs）、两种数据集（改编和合成）、两种组织结构（Basic 和 Office），并测试了多种威胁场景（如提示注入、角色不匹配、越狱攻击）。测试集设计平衡（50% 正例和负例），包含未见指令和改写指令，确保评估全面。
*   **局限性与对比:** 在细粒度角色区分和对抗性攻击（如破损角色格式）上准确率下降 15-30%，Office 结构（复杂层次）下的性能普遍低于 Basic 结构，显示模型对复杂层次结构的泛化能力不足。Role-aware LLM-Gen 在破损角色攻击防御上表现较好（平均准确率 56%），优于其他方法。

## Further Thoughts

论文中角色编码策略对模型性能的影响是一个值得关注的启发点。实验表明，基于名称的编码（如 Hierarchical Name Encoding）在区分授权角色时优于基于数字的编码，但对对抗性角色的鲁棒性较差。这启发我们思考如何设计更鲁棒的角色表示方法，例如结合语义和结构信息，或通过多适配器策略（每个部门一个适配器）进一步隔离知识，减少信息泄露风险。此外，探索动态角色更新机制（在微调后添加或修改角色）可能是一个有前景的方向。