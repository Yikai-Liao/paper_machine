---
title: "LeanK: Learnable K Cache Channel Pruning for Efficient Decoding"
pubDatetime: 2025-08-04T09:08:43+00:00
slug: "2025-08-leank-kcache-pruning"
type: "arxiv"
id: "2508.02215"
score: 0.7127012493622015
author: "grok-3-latest"
authors: ["Yike Zhang", "Zhiyuan He", "Huiqiang Jiang", "Chengruidong Zhang", "Yuqing Yang", "Jianyong Wang", "Lili Qiu"]
tags: ["LLM", "KV Cache", "Pruning", "Long Context", "Efficiency"]
institution: ["Tsinghua University", "Microsoft Research"]
description: "本文提出 LeanK，一种基于学习的静态剪枝方法，通过双阶段训练优化 K Cache 通道剪枝，显著减少长上下文 LLM 推理的内存占用和延迟，同时保持模型性能。"
---

> **Summary:** 本文提出 LeanK，一种基于学习的静态剪枝方法，通过双阶段训练优化 K Cache 通道剪枝，显著减少长上下文 LLM 推理的内存占用和延迟，同时保持模型性能。 

> **Keywords:** LLM, KV Cache, Pruning, Long Context, Efficiency

**Authors:** Yike Zhang, Zhiyuan He, Huiqiang Jiang, Chengruidong Zhang, Yuqing Yang, Jianyong Wang, Lili Qiu

**Institution(s):** Tsinghua University, Microsoft Research


## Problem Background

大型语言模型（LLMs）在长上下文任务中因键值缓存（KV Cache）的高内存占用和推理延迟面临效率挑战。
现有优化方法（如逐出、选择、量化）忽略了键缓存（K Cache）通道维度的稀疏性潜力，而这种稀疏性具有静态特性，为离线剪枝提供了可能性。
论文旨在通过剪枝不重要的 K Cache 通道，减少 GPU 内存使用并加速解码，同时尽量保持模型性能。

## Method

*   **核心思想:** 提出 LeanK，一种基于学习的静态剪枝方法，通过双阶段训练过程学习 K Cache 通道的剪枝掩码，以优化长上下文解码效率。
*   **第一阶段 - 重要性学习:** 引入一个连续的缩放因子（scaling factor α）表示每个通道的全局重要性，使用 L2 蒸馏损失（确保剪枝后隐藏状态接近原始状态）和 L1 正则化损失（鼓励稀疏性）进行训练。训练数据基于检索任务（如密集检索和多值检索），重点优化解码阶段的性能，特别针对长上下文的中部注意力区域进行缩放，而保留关键的注意力沉点和局部窗口。
*   **第二阶段 - 掩码优化:** 将连续缩放因子转换为二值掩码（β），满足预定义剪枝比例（如 70%）和硬件对齐要求（如通道数为 16 或 32 的倍数）。通过进一步训练（仅使用蒸馏损失）调整掩码，减少性能损失，确保部署时的效率。
*   **部署策略:** 在推理时，根据学习到的静态掩码剪枝 K Cache，结合结构化注意力计算（保留沉点和局部窗口的完整缓存，剪枝中部区域），并每 32 个 token 更新一次缓存以减少开销。对于完全剪枝的注意力头，可进一步省略对应的值缓存（V Cache）。
*   **关键点:** 方法不依赖动态计算，静态掩码离线学习，避免推理时额外开销；同时与硬件优化（如自定义解码内核）结合，提升计算效率。

## Experiment

*   **有效性:** 在 Llama-3.1-8B-Instruct 和 Qwen2.5-7B-Instruct 模型上，LeanK 以 70% 剪枝比例实现 K Cache 内存减少约 70%，V Cache 减少 16%-18%，注意力计算速度提升 1.3x（Llama）至 1.6x（Qwen），端到端吞吐量提升 1.2x。性能损失极小，例如在 RULER 基准上仅下降 0.3%（Llama），而对比方法 ThinK 下降高达 52.8%。
*   **全面性与合理性:** 实验覆盖 LongBench、RULER 和 GSM-Infinite 三个基准，测试上下文长度从 4K 到 128K，任务类型包括问答、推理、代码生成等，验证了静态剪枝策略在多样场景下的稳定性。训练和测试设置合理，针对不同模型（如 Qwen 的 Yarn 扩展）调整剪枝掩码，确保适应性。
*   **与其他方法对比:** LeanK 优于动态剪枝方法 ThinK 和 token 级剪枝方法 SnapKV，尤其在复杂检索任务中表现更佳；与现有优化方法（如 DuoAttention、Quest、KIVI）结合后进一步提升效率，例如与 KIVI 结合将 KV Cache 压缩比从 5.3x 提升到 9.7x。
*   **局限性:** 实验未深入探讨超长上下文（>128K）或低资源设备下的表现，可能是未来改进方向。

## Further Thoughts

LeanK 揭示了 RoPE 位置编码对 K Cache 通道重要性的影响，低频通道更关键，这启发是否可以通过改进位置编码设计减少冗余；静态剪枝的高效性提示在预训练阶段引入稀疏性约束的潜力；此外，V Cache 剪枝潜力和任务特定剪枝策略（如针对检索 vs 生成）也值得探索，或许可以结合动态调整与静态掩码，针对性优化不同场景。