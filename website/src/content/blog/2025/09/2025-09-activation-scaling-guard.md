---
title: "ASGuard: Activation-Scaling Guard to Mitigate Targeted Jailbreaking Attack"
pubDatetime: 2025-09-30T06:33:52+00:00
slug: "2025-09-activation-scaling-guard"
type: "arxiv"
id: "2509.25843"
score: 0.7008805820750725
author: "grok-3-latest"
authors: ["Yein Park", "Jungwoo Park", "Jaewoo Kang"]
tags: ["LLM", "Safety Alignment", "Mechanistic Interpretability", "Attention Heads", "Fine-Tuning"]
institution: ["Korea University", "AIGEN Sciences"]
description: "本文提出ASGuard框架，通过机制可解释性识别时态越狱漏洞并进行精准激活缩放和预防性微调，显著降低攻击成功率，同时在安全与实用性之间实现Pareto最优平衡。"
---

> **Summary:** 本文提出ASGuard框架，通过机制可解释性识别时态越狱漏洞并进行精准激活缩放和预防性微调，显著降低攻击成功率，同时在安全与实用性之间实现Pareto最优平衡。 

> **Keywords:** LLM, Safety Alignment, Mechanistic Interpretability, Attention Heads, Fine-Tuning

**Authors:** Yein Park, Jungwoo Park, Jaewoo Kang

**Institution(s):** Korea University, AIGEN Sciences


## Problem Background

大型语言模型（LLMs）尽管经过安全对齐训练，但在面对时态越狱攻击（tense jailbreaking）时表现出脆弱性，即通过简单的语义变换（如将现在时改为过去时）即可绕过拒绝机制，揭示了当前对齐方法在语义泛化上的关键不足。
这一问题不仅威胁模型安全，还可能导致有害内容的生成，亟需一种针对性解决方案来修补特定漏洞，同时避免影响模型的通用能力。

## Method

*   **核心思想:** 提出Activation-Scaling Guard (ASGuard)，一个基于机制可解释性的多阶段框架，通过精准干预模型内部组件，针对性地修补时态越狱漏洞，同时保持安全与实用性的平衡。
*   **步骤一 - 电路分析（Circuit Analysis）:** 利用边缘归因补丁（Edge Attribution Patching with Integrated Gradients, EAP-IG）方法，构建目标漏洞电路，识别与时态越狱攻击因果相关联的特定注意力头（attention heads）。通过对比成功越狱（过去时）和失败越狱（现在时）的电路，筛选出仅在成功案例中活跃的‘时态漏洞头’。
*   **步骤二 - 激活缩放（Activation Scaling）:** 针对识别出的漏洞注意力头，采用‘识别后缩放’（Identify-then-Scale）协议，训练一个通道级别的缩放向量（scaling vector），通过元素级乘法调整其激活值，抑制有害信息路径的传播，同时避免完全消融导致的功能损失。
*   **步骤三 - 预防性微调（Preventative Fine-Tuning）:** 在缩放向量辅助下，使用拒绝行为数据集对模型进行微调，引导其学习更鲁棒的拒绝机制；随后移除缩放向量，确保模型通过优化找到替代路径，形成内在的安全能力，而非依赖外部干预。
*   **关键优势:** 方法具有参数高效性（仅训练少量缩放参数，模型权重冻结），且干预精准，避免了传统对齐方法可能导致的过度拒绝或能力下降；最终模型在推理时无额外计算成本。

## Experiment

*   **有效性:** ASGuard在三个开源模型（Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9b-it）上显著降低了时态越狱攻击成功率（ASR），例如Llama从42%降至8%，Qwen从51%降至8%，Gemma从38%降至19%，优于单纯激活缩放或头部消融的效果。
*   **安全-实用性平衡:** 与基线方法（如SFT, DPO）相比，ASGuard在Pareto前沿上表现更优，避免了过度拒绝（over-refusal）和灾难性遗忘（catastrophic forgetting）。例如，SFT在Qwen上虽将ASR降至0%，但过度拒绝率高达98.5%，而ASGuard的R-Score（综合安全与实用性指标）保持较高（如Llama为71.8）。
*   **实验设置合理性:** 实验设计全面，涵盖目标拒绝、一般拒绝、过度拒绝和通用能力（MMLU）四个维度，数据集（如JBB-Behaviors, OR-Bench）具有代表性，评估指标（如ASR, R-Score）科学合理；同时使用GPT-4.1作为语义评判模型，确保结果可信。
*   **局限性:** 论文指出方法对基于蒸馏或MoE架构的模型可能不直接适用，小型模型对注意力头干预也过于敏感，提示通用性需进一步验证。

## Further Thoughts

ASGuard通过机制可解释性精准定位漏洞并进行干预的思路令人启发，是否可以扩展到其他语义攻击（如情感操控或逻辑越狱）？此外，‘临时干预+永久学习’的模式是否适用于去偏见或知识更新等领域？论文揭示的注意力头功能分离（时态处理、危害性评估、拒绝机制）也提示我们，是否可以设计模块化的安全机制，分别优化不同功能组件？