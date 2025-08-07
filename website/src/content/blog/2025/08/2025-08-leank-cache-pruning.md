---
title: "LeanK: Learnable K Cache Channel Pruning for Efficient Decoding"
pubDatetime: 2025-08-04T09:08:43+00:00
slug: "2025-08-leank-cache-pruning"
type: "arxiv"
id: "2508.02215"
score: 0.7127012493622015
author: "grok-3-latest"
authors: ["Yike Zhang", "Zhiyuan He", "Huiqiang Jiang", "Chengruidong Zhang", "Yuqing Yang", "Jianyong Wang", "Lili Qiu"]
tags: ["LLM", "KV Cache", "Pruning", "Efficiency", "Long Context"]
institution: ["Tsinghua University", "Microsoft Research"]
description: "LeanK 提出了一种基于学习的静态 K Cache 通道剪枝方法，通过双阶段训练过程显著减少长上下文推理中的内存占用并加速计算，同时保持模型精度。"
---

> **Summary:** LeanK 提出了一种基于学习的静态 K Cache 通道剪枝方法，通过双阶段训练过程显著减少长上下文推理中的内存占用并加速计算，同时保持模型精度。 

> **Keywords:** LLM, KV Cache, Pruning, Efficiency, Long Context

**Authors:** Yike Zhang, Zhiyuan He, Huiqiang Jiang, Chengruidong Zhang, Yuqing Yang, Jianyong Wang, Lili Qiu

**Institution(s):** Tsinghua University, Microsoft Research


## Problem Background

大型语言模型（LLMs）在长上下文任务中面临效率挑战，键值缓存（KV Cache）随着上下文长度增加而显著扩大，导致GPU内存占用和推理延迟急剧上升。
现有优化方法（如逐出、选择和量化）未能充分利用键缓存（K Cache）通道维度的稀疏性，而论文发现部分 K 通道重要性较低且具有静态特性，提出通过剪枝这些通道来提升内存效率和推理速度。

## Method

*   **核心思想:** 提出 LeanK，一种基于学习的 K Cache 通道剪枝方法，通过静态稀疏性减少内存占用和加速长上下文解码，同时保持模型性能。
*   **双阶段训练过程:**
    *   **第一阶段:** 学习每个 K 通道的全局重要性分数（通过连续缩放因子 α），使用 L2 蒸馏损失确保缩放后的隐藏状态接近原始状态，并引入 L1 正则化损失促进稀疏性；训练数据包括密集检索和多值检索任务，关注解码阶段性能。
    *   **第二阶段:** 将连续缩放因子转化为二值掩码 β，满足预定义剪枝比例和硬件对齐要求（如通道数为 16 或 32 的倍数），仅使用蒸馏损失以维持模型精度。
*   **部署策略:** 在推理时，根据学习到的静态掩码剪枝 K Cache，结合注意力沉降（Attention Sink）和局部窗口机制，优化注意力计算，减少内存带宽使用；对于完全剪枝的注意力头，可进一步节省对应的 V Cache。
*   **关键创新:** 静态剪枝策略区别于动态方法，注重硬件效率，且与现有 KV Cache 优化方法正交，可联合使用。

## Experiment

*   **有效性:** 在 Llama-3.1-8B-Instruct 和 Qwen2.5-7B-Instruct 模型上，LeanK 以 70% 剪枝比例实现 K Cache 内存减少约 70%，V Cache 减少 16%-18%，模型精度几乎无损（例如在 RULER 基准上仅下降 0.3% 和 0.1%），显著优于动态方法 ThinK（在相同比例下 Llama 性能下降 52.8%）。
*   **速度提升:** 定制解码内核实现注意力计算 1.3x-1.6x 加速，端到端吞吐量提升 1.2x，验证了方法在实际推理中的效率优势。
*   **泛化性与鲁棒性:** 静态通道重要性模式在不同输入长度（4K-128K）和任务类型（LongBench, RULER, GSM-Infinite）上表现稳定，尤其在长生成推理任务中优于 ThinK。
*   **正交性验证:** 与量化（KIVI）、选择性读取（Quest）和逐出（DuoAttention）方法结合，进一步提升压缩比（例如与 KIVI 结合从 5.3x 提升到 9.7x）。
*   **实验设置:** 覆盖多种上下文长度、任务类型和模型，数据对比清晰，但未探讨极端长上下文（>128K）或更多模型架构的表现。

## Further Thoughts

LeanK 揭示了 K Cache 通道重要性的静态特性，启发我们探索其他模型组件（如权重或激活值）中的静态稀疏性潜力；此外，低频通道在长上下文理解中的重要性提示可以在位置编码设计或预训练阶段优化频率分布，减少冗余；同时，LeanK 与其他优化方法的正交性表明未来可构建多层次优化框架，结合剪枝、量化和逐出策略，针对不同硬件环境自适应调整。