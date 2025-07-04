---
title: "Reasoning as an Adaptive Defense for Safety"
pubDatetime: 2025-07-01T17:20:04+00:00
slug: "2025-07-adaptive-reasoning-safety"
type: "arxiv"
id: "2507.00971"
score: 0.6348786952456904
author: "grok-3-latest"
authors: ["Taeyoun Kim", "Fahim Tajwar", "Aditi Raghunathan", "Aviral Kumar"]
tags: ["LLM", "Reasoning", "Test Time Scaling", "Post-Training", "RLHF"]
institution: ["Carnegie Mellon University"]
description: "本文提出 TARS 方法，通过强化学习和自适应推理训练大型语言模型动态判断提示安全性，显著提升了安全-拒绝权衡并增强了对越狱攻击的鲁棒性。"
---

> **Summary:** 本文提出 TARS 方法，通过强化学习和自适应推理训练大型语言模型动态判断提示安全性，显著提升了安全-拒绝权衡并增强了对越狱攻击的鲁棒性。 

> **Keywords:** LLM, Reasoning, Test Time Scaling, Post-Training, RLHF

**Authors:** Taeyoun Kim, Fahim Tajwar, Aditi Raghunathan, Aviral Kumar

**Institution(s):** Carnegie Mellon University


## Problem Background

大型语言模型（LLMs）在面对安全漏洞（如越狱攻击）时表现出脆弱性，现有防御策略多为静态保护，难以适应复杂或模糊的有害请求。
论文旨在探索如何通过测试时计算（test-time compute）进行自适应推理，让模型在每个提示的基础上动态判断安全性，从而在安全性和任务完成之间取得更好的平衡。

## Method

*   **核心思想:** 提出 TARS（Training Adaptive Reasoners for Safety），一种通过强化学习（RL）和长链式推理（Chain-of-Thought, CoT）训练模型自适应判断提示安全性的方法。
*   **具体实现:** 
    *   **阶段一 - 轻量级监督微调（SFT）:** 在基础模型（如 Qwen-2.5-1.5B-Instruct）上，使用低学习率和少量训练轮数进行微调，数据包括有害提示及其从 DeepSeek-R1 模型提取的推理轨迹，旨在增加生成多样性，为后续 RL 提供探索基础。
    *   **阶段二 - 提示集设计:** 构建 RL 训练数据，混合有害提示（来自 WildJailbreak、Aegis 等）、无害提示（来自 UltraFeedback）和模糊提示（来自 OR-Bench），避免模型学习过度拒绝的捷径行为，同时通过彩虹团队（Rainbow Teaming）生成对抗性提示增强数据多样性。
    *   **阶段三 - 强化学习（RL）训练:** 采用 GRPO 方法，设计复合奖励函数，包括安全奖励（基于 Moderation API 评估有害性）、任务完成奖励（基于 GRM 模型评估帮助性）和格式奖励（确保推理格式包含 `<think>` 和 `</think>` 标记）。训练过程中，模型学习在有害和无害提示间动态平衡安全性和帮助性，同时保持推理能力。
*   **关键特点:** 不修改基础模型结构，仅通过后训练（post-training）调整推理行为；奖励分离设计避免了推理能力退化；自适应计算使模型根据提示复杂性调整推理长度。

## Experiment

*   **有效性:** TARS 在安全-拒绝权衡（safety-refusal trade-off）上显著优于 SFT、DPO 和无推理的 RL 方法，在 Harmbench 上的防御成功率（DSR）最高达 92.79%（GCG 攻击下，格式正确时），而对比方法如 SFT 仅为 82.73%。
*   **自适应性:** 在 Sorry-Bench 数据集上，TARS 模型根据提示复杂性调整推理长度，例如对‘不合格建议’类提示的推理长度最长（456.66  token），表明其能动态分配计算资源。
*   **鲁棒性:** TARS 对白盒攻击（如 GCG）和黑盒攻击（如 PAIR）均表现出更强鲁棒性，即使推理格式被破坏，DSR 仍高于 SFT（83.82% vs 78.90%）。与开源模型（如 Llama-RR 8B）相比，TARS 在更小规模（1.5B 参数）下实现了更好的安全-拒绝平衡。
*   **实验设置合理性:** 实验涵盖了多种攻击类型（GCG、PAIR、AutoDAN、PAP）、不同提示混合比例（λ=0.1至0.9）和多个基准测试（Harmbench、XSTest、Sorry-Bench），设置较为全面；但缺乏对更大规模模型（如 70B 参数）效果的测试，可能限制了结果的普适性。
*   **开销:** 主要增加在于 RL 训练阶段的计算成本（如多轮生成和奖励计算），但推理时仅需额外格式标记和自适应计算，成本可控。

## Further Thoughts

TARS 通过自适应推理提升安全性的思路启发了我，是否可以进一步探索动态推理结构的多样性（如根据提示类型调整推理深度或引入多层推理），以应对更复杂的攻击场景？此外，奖励分离设计让我思考是否可以通过多目标优化或上下文感知的动态奖励机制，进一步提升模型在不同用户场景下的适应性。