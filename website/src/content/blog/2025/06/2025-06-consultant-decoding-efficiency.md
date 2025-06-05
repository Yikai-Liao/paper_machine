---
title: "Consultant Decoding: Yet Another Synergistic Mechanism"
pubDatetime: 2025-06-03T03:13:27+00:00
slug: "2025-06-consultant-decoding-efficiency"
type: "arxiv"
id: "2506.02391"
score: 0.6913697198555762
author: "grok-3-latest"
authors: ["Chuanghao Ding", "Jiaping Wang", "Ziqing Yang", "Xiaoliang Wang", "Dahua Lin", "Cam-Tu Nguyen", "Fei Tan"]
tags: ["LLM", "Speculative Decoding", "Inference Efficiency", "Draft Model", "Token Verification"]
institution: ["State Key Laboratory for Novel Software Technology, Nanjing University", "East China Normal University", "Independent Researcher", "The Chinese University of Hong Kong"]
description: "本文提出顾问解码（CD），一种基于负对数似然的验证机制，显著提升大型语言模型推理速度达 3.09 倍，同时保持甚至超越目标模型性能。"
---

> **Summary:** 本文提出顾问解码（CD），一种基于负对数似然的验证机制，显著提升大型语言模型推理速度达 3.09 倍，同时保持甚至超越目标模型性能。 

> **Keywords:** LLM, Speculative Decoding, Inference Efficiency, Draft Model, Token Verification

**Authors:** Chuanghao Ding, Jiaping Wang, Ziqing Yang, Xiaoliang Wang, Dahua Lin, Cam-Tu Nguyen, Fei Tan

**Institution(s):** State Key Laboratory for Novel Software Technology, Nanjing University, East China Normal University, Independent Researcher, The Chinese University of Hong Kong


## Problem Background

大型语言模型（LLMs）因其卓越性能和泛化能力被广泛应用，但随着模型规模的增长（如 LLaMA3.1-405B），推理效率成为关键瓶颈。
传统的推测解码（Speculative Decoding, SD）通过小型草稿模型生成候选 token 并由大型目标模型验证来加速推理，但其基于重要性采样的验证机制导致草稿 token 拒绝率高，频繁调用目标模型，效率提升受限。
论文旨在设计一种更高效的验证机制，以提高 token 接受率，减少目标模型调用，同时维持生成质量。

## Method

*   **核心思想:** 提出顾问解码（Consultant Decoding, CD），通过目标模型的负对数似然（Negative Log-Likelihood, NLL）直接评估草稿 token 的正确性，而非依赖似然比，确保高效验证并保持生成质量。
*   **具体实现步骤:**
    *   小型草稿模型（Draft Model）以自回归方式生成一组候选 token（长度为 γ）。
    *   大型目标模型（Target Model）并行计算这些 token 的概率分布，得到每个 token 的 NLL。
    *   验证规则：若某个 token 的 NLL 低于预设阈值 ε（基于目标模型训练时的收敛损失，通用值为 2.0），则接受该 token；否则，从目标模型的分布中重新采样一个新 token，并丢弃后续草稿 token。
    *   平滑验证机制：引入指数移动平均（EMA）平衡当前 token 和上下文 token 的 NLL 贡献，通过参数 β 调节上下文影响，避免偶然错误。
*   **阈值确定:** 阈值 ε 通过 Chinchilla 缩放法则估算目标模型的收敛损失，设置为 2.0，无需额外调参，适用于多种模型和任务。
*   **设计理念:** CD 模拟‘顾问’模式，草稿模型自主生成，目标模型提供验证建议，兼顾独立性和协作性，避免严格分布一致性带来的高拒绝率。

## Experiment

*   **速度提升:** CD 在多个任务（如 GSM8K, HumanEval, MT-Bench, AlpacaEval）上实现了显著加速，最高达 3.09 倍（Qwen2.5-0.5B/72B 在 GSM8K 上），相比 SD 和 Mentored Decoding (MD) 分别快 0.8 倍和 0.59 倍。
*   **调用减少:** CD 将目标模型调用比例降至最低 9.1%（Qwen2.5-0.5B/72B 在 GSM8K 和 HumanEval 上），远低于 SD 和 MD，显著提升并行任务吞吐量。
*   **生成质量:** CD 保持目标模型性能的约 100%，在某些场景（如 HumanEval 上 Qwen2.5-7B/72B）甚至提升 3.5%，超出推测解码的理论上限。
*   **鲁棒性:** CD 对草稿长度变化表现出更强适应性，加速比下降幅度（0.08-0.19 倍）远低于 SD（0.53-0.59 倍）和 MD（0.32-0.4 倍）。
*   **实验设置合理性:** 实验覆盖数学推理、代码生成、对话和指令跟随等多种任务，模型组合包括参数规模相差两个数量级的搭配（如 0.5B/72B），同时在独立草稿和自草稿设置下验证了方法的普适性；对比了通用设置和速度优化设置，数据支持 CD 在效率和质量上的优越平衡。

## Further Thoughts

CD 启发我们探索更多非分布一致性的验证机制，如基于语义相似度或任务特定指标的验证；小模型引导大模型推理路径的可能性提示未来可研究小模型如何在特定任务上‘启发’大模型；固定阈值 ε 的局限性启发动态阈值优化的方向，根据任务、模型组合或上下文自适应调整阈值以进一步提升性能。