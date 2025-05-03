---
title: "The Illusion of Role Separation: Hidden Shortcuts in LLM Role Learning (and How to Fix Them)"
pubDatetime: 2025-05-01T16:06:16+00:00
slug: "2025-05-role-separation-learning"
type: "arxiv"
id: "2505.00626"
score: 0.7648768928079644
author: "grok-3-latest"
authors: ["Zihao Wang", "Yibo Jiang", "Jiahao Yu", "Heqing Huang"]
tags: ["LLM", "Role Separation", "Fine-Tuning", "Position Encoding", "Prompt Injection"]
institution: ["University of Chicago", "Northwestern University", "ByteDance Inc."]
description: "本文通过操纵位置 ID 增强大型语言模型的角色分离能力，提出位置增强微调（PFT）方法，显著缓解模型对任务类型和文本开头位置的捷径依赖，同时维持常规任务性能。"
---

> **Summary:** 本文通过操纵位置 ID 增强大型语言模型的角色分离能力，提出位置增强微调（PFT）方法，显著缓解模型对任务类型和文本开头位置的捷径依赖，同时维持常规任务性能。 

> **Keywords:** LLM, Role Separation, Fine-Tuning, Position Encoding, Prompt Injection

**Authors:** Zihao Wang, Yibo Jiang, Jiahao Yu, Heqing Huang

**Institution(s):** University of Chicago, Northwestern University, ByteDance Inc.


## Problem Background

大型语言模型（LLMs）在处理多角色输入（如系统指令、用户查询）时，需准确区分各角色信息以确保一致的行为和安全性。然而，现有方法可能仅通过记忆攻击模式（如提示注入攻击）而非真正理解角色边界来应对挑战，这导致功能性失败和潜在的安全漏洞。

## Method

* **核心思想：** 研究模型在角色分离学习中的根本问题，并提出增强角色边界不变信号的方法，避免模型依赖捷径（如任务类型关联和靠近文本开头的优先级）。
* **实验框架：** 设计闭域设置，将系统输入视为指令，用户输入视为数据，使用‘良性’训练数据和‘对抗性’评估数据，隔离模式匹配的影响，测试真正的角色区分能力。
* **捷径识别与缓解：** 发现模型依赖两种捷径：任务类型关联（即根据任务类型而非角色决定行为）和靠近文本开头（即优先处理靠近输入起始的令牌）。通过数据增强（如交换系统和用户内容、插入非关键信息）缓解这些捷径，但指出这种方法只是临时修补。
* **位置增强微调（PFT）：** 提出通过操纵位置 ID 在系统和用户令牌间创建固定差距（例如 d=256 或 512），同时保持角色内部顺序，增强角色区分信号。模型在微调时学习利用这一信号，正确对待系统令牌为指令，用户令牌为数据。
* **实现细节：** 使用标准监督微调（SFT）和低秩适应（LoRA）技术，基于 Llama-3-8B-Instruct 和 Gemma-2-9b-it 模型进行实验，确保方法的可行性和泛化性。

## Experiment

* **有效性：** PFT 方法在对抗性评估（如 Gandalf Summarization, TensorTrust Extraction）上显著优于标准 SFT，例如在 TensorTrust Extraction 攻击中，Llama 模型准确率从 33% 提升至 62%，Gemma 模型从 70% 提升至 92%，表明角色分离能力得到增强。
* **全面性与合理性：** 实验设置覆盖多种攻击类型，并通过普通数据评估（如 Alpaca 数据集）验证 PFT 维持了模型在常规任务上的性能（准确率和生成质量无显著下降）。此外，测试不同位置差距参数（d=256, 512）以验证方法的鲁棒性。
* **局限性：** 尽管 PFT 表现优于基线，但在某些攻击（如 TensorTrust Hijacking）上提升有限（Llama 模型从 33% 提升至 37%），且未探讨开放域或更复杂的多角色场景。

## Further Thoughts

论文中通过位置 ID 操纵增强角色区分信号的思路非常启发性，提示我们可以在输入编码的其他维度（如令牌嵌入或注意力机制）引入角色特定信号；此外，模型对预训练机制（如注意力沉没现象）的依赖表明，可以在预训练阶段设计角色区分相关的训练目标，以从根本上避免捷径学习问题。