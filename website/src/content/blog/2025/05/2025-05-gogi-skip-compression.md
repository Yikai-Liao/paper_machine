---
title: "Accelerating Chain-of-Thought Reasoning: When Goal-Gradient Importance Meets Dynamic Skipping"
pubDatetime: 2025-05-13T09:39:18+00:00
slug: "2025-05-gogi-skip-compression"
type: "arxiv"
id: "2505.08392"
score: 0.5957462606139421
author: "grok-3-latest"
authors: ["Ren Zhuang", "Ben Wang", "Shuifa Sun"]
tags: ["LLM", "Chain of Thought", "Compression", "Reasoning", "Dynamic Adaptation"]
institution: ["未明确列出具体机构，推测为学术或科技研究单位"]
description: "本文提出 Adaptive GoGI-Skip 框架，通过目标梯度重要性（GoGI）和动态跳跃（ADS）实现 CoT 压缩，减少超 45% token 并提升 1.6x-2.0x 推理速度，同时保持高准确性，超越静态压缩方法。"
---

> **Summary:** 本文提出 Adaptive GoGI-Skip 框架，通过目标梯度重要性（GoGI）和动态跳跃（ADS）实现 CoT 压缩，减少超 45% token 并提升 1.6x-2.0x 推理速度，同时保持高准确性，超越静态压缩方法。 

> **Keywords:** LLM, Chain of Thought, Compression, Reasoning, Dynamic Adaptation

**Authors:** Ren Zhuang, Ben Wang, Shuifa Sun

**Institution(s):** 未明确列出具体机构，推测为学术或科技研究单位


## Problem Background

大型语言模型（LLMs）通过链式思维（Chain-of-Thought, CoT）提示在复杂任务中表现出色，但生成的推理轨迹往往冗长且低效，导致高计算成本、延迟和内存需求，成为实际应用的瓶颈。
现有 CoT 压缩方法依赖通用重要性指标和静态压缩率，可能会错误删除功能关键的 token 或无法适应推理复杂度的变化，论文旨在解决如何动态高效压缩 CoT 序列，同时保持推理准确性。

## Method

*   **核心思想:** 通过监督微调（Supervised Fine-Tuning, SFT）学习动态压缩 CoT 序列，结合目标导向的重要性评估和运行时不确定性感知的跳跃策略，实现高效推理。
*   **具体实现:** 提出 Adaptive GoGI-Skip 框架，包含两个创新组件：
    *   **Goal-Gradient Importance (GoGI):** 一种新颖的重要性度量方法，通过计算每个 token 的中间表示对最终答案损失（answer loss）的梯度影响，精准识别功能上关键的 token，直接关联推理目标，而非依赖语义相似度或困惑度等通用指标。
    *   **Adaptive Dynamic Skipping (ADS):** 动态调整压缩率的机制，包含两个子模块：
        *   **Entropy-Driven Rate (EDR) Regulation:** 基于运行时预测熵（uncertainty）调整 token 保留率，高熵（高不确定性）时保守压缩，保留更多 token；低熵时激进压缩，减少冗余。
        *   **Adaptive N-Constraint (ANC):** 通过局部上下文复杂性（基于窗口熵）动态限制连续删除的 token 数量，确保推理连贯性，防止关键结构信息丢失。
*   **关键点:** 该方法在离线阶段计算 GoGI 分数并生成压缩数据用于 SFT，不需在推理时增加额外参数，通过动态调整兼顾效率与准确性。

## Experiment

*   **有效性:** 在多个基准数据集（AIME, GPQA, GSM8K）上，Adaptive GoGI-Skip 平均减少超过 45% 的 CoT token 数量，推理速度提升 1.6x 至 2.0x，同时在大多数任务上保持与原始模型接近的准确率（例如 GSM8K 上提升 0.3%）。
*   **优越性:** 相比静态压缩基线（如 TokenSkip），该方法在高压缩率下仍维持准确性，避免了显著性能下降，例如在 AIME’25 上仅下降 0.9%，而 TokenSkip（γ=0.5）下降 3.4%。
*   **实验设置合理性:** 实验覆盖多种模型规模（1B 至 12B 参数）和架构（Gemma3-Instruct, Qwen2.5-Instruct），数据基于 MATH 数据集（7472 样本），对比多种基线（Original, Prompting, C3ot, Spiritft, TokenSkip），并通过消融研究验证各组件贡献，设置全面且结果稳定（准确率取 3 次微调平均）。
*   **开销:** GoGI 分数计算为离线过程，需额外计算梯度（例如 Gemma-3-4B-it 上约 12 GPU 小时），但推理时无额外参数负担，整体成本可控。

## Further Thoughts

GoGI 指标通过梯度影响直接衡量 token 对推理目标的贡献，这种目标导向评估可推广至其他 NLP 任务，用于关键信息识别或模型解释；ADS 的动态调整机制（结合熵和局部约束）启发在自适应计算场景中应用不确定性驱动策略，如资源分配或推理深度控制；此外，论文提到的 RL 端到端学习跳跃策略提示未来可探索 SFT 与 RL 混合训练，进一步提升效率。