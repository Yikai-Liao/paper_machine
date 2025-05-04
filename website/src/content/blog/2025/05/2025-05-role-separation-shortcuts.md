---
title: "The Illusion of Role Separation: Hidden Shortcuts in LLM Role Learning (and How to Fix Them)"
pubDatetime: 2025-05-01T16:06:16+00:00
slug: "2025-05-role-separation-shortcuts"
type: "arxiv"
id: "2505.00626"
score: 0.7648768928079644
author: "grok-3-latest"
authors: ["Zihao Wang", "Yibo Jiang", "Jiahao Yu", "Heqing Huang"]
tags: ["LLM", "Role Separation", "Position Encoding", "Fine-Tuning", "Prompt Injection"]
institution: ["University of Chicago", "Northwestern University", "ByteDance Inc."]
description: "本文提出位置增强微调（PFT）方法，通过操纵位置 ID 增强角色分离信号，有效缓解大型语言模型对任务类型和位置捷径的依赖，同时保持性能。"
---

> **Summary:** 本文提出位置增强微调（PFT）方法，通过操纵位置 ID 增强角色分离信号，有效缓解大型语言模型对任务类型和位置捷径的依赖，同时保持性能。 

> **Keywords:** LLM, Role Separation, Position Encoding, Fine-Tuning, Prompt Injection

**Authors:** Zihao Wang, Yibo Jiang, Jiahao Yu, Heqing Huang

**Institution(s):** University of Chicago, Northwestern University, ByteDance Inc.


## Problem Background

大型语言模型（LLMs）在处理多角色输入（如系统指令、用户查询）时，需要准确区分各角色信息以确保功能性和安全性，这一能力称为角色分离（Role Separation）。
然而，现有研究多关注提示注入攻击（Prompt Injection Attacks）的防御，未深入探讨模型是否真正学会角色区分，还是仅依赖表面捷径（如任务类型关联和靠近文本开头的优先级），这导致模型在面对新型攻击或复杂提示时易失效，存在功能错误和安全隐患。

## Method

*   **核心思想:** 通过增强角色之间的不变信号（Invariant Signals），让模型真正学会区分系统和用户角色，而非依赖任务类型或位置捷径。
*   **具体实现:** 提出位置增强微调（Position-enhanced Fine-Tuning, PFT）方法，通过操纵位置 ID（Position IDs）来增强角色区分：
    *   在系统角色和用户角色的位置 ID 之间引入固定距离（Gap），例如系统角色最后一个 token 位置为 k，则用户角色第一个 token 位置设为 k+1+d（d 为预设距离），以创造数值边界。
    *   保持每个角色内部 token 的相对顺序不变，确保模型对序列关系的理解不受影响。
    *   在此基础上进行监督微调（Supervised Fine-Tuning, SFT），使用 LoRA 适配技术优化模型对目标响应的对数概率。
*   **实验框架:** 设计受控实验框架，使用‘良性’训练数据和‘对抗性’评估数据，隔离模式匹配与真正角色学习的影响，识别模型依赖的捷径。
*   **关键点:** PFT 是一种 token 级别的信号增强方法，相较于仅依赖分隔符 token 或数据增强，能更普适地应对复杂提示结构，且不损害模型原有性能。

## Experiment

*   **有效性:** PFT 方法显著提升了角色分离能力，在对抗性评估（如 Gandalf Summarization, TensorTrust Extraction）中表现优异，例如在 TensorTrust Extraction 攻击上，Llama 模型准确率从 33% 提升至 62%，Gemma 模型从 70% 提升至 92%，表明有效缓解了任务类型和靠近文本开头捷径的影响。
*   **性能稳定性:** 在普通数据（如 Alpaca 数据集）上的准确率和生成质量（以对数似然度量）与标准监督微调（SFT）相当，KL 散度显示 PFT 模型与基础模型分布偏差极小，证明位置 ID 操纵未损害模型性能。
*   **实验设置合理性:** 实验在 Llama-3-8B-Instruct 和 Gemma-2-9b-it 模型上进行，涵盖多种攻击类型和普通数据评估，受控实验框架避免了模型单纯记忆攻击模式，增强结论普适性和可信度；但未探讨超长上下文场景下 PFT 效果，可能为潜在局限。

## Further Thoughts

增强不变信号（Invariant Signals）的思路可扩展至其他需要明确边界区分的任务，如多轮对话中历史上下文与当前输入的分离，或多模态模型中文本与图像信息的区分；此外，预训练模型对初始 token 的‘注意力沉积’（Attention Sink）现象提示我们，可在预训练阶段引入角色区分信号，而非仅依赖后训练微调，或设计角色特定嵌入和动态位置编码以应对复杂输入结构。