---
title: "MixAT: Combining Continuous and Discrete Adversarial Training for LLMs"
pubDatetime: 2025-05-22T17:32:50+00:00
slug: "2025-05-mixat-adversarial-training"
type: "arxiv"
id: "2505.16947"
score: 0.6389845361052162
author: "grok-3-latest"
authors: ["Csaba Dékány", "Stefan Balauca", "Robin Staab", "Dimitar I. Dimitrov", "Martin Vechev"]
tags: ["LLM", "Adversarial Training", "Robustness", "Continuous Attacks", "Discrete Attacks"]
institution: ["INSAIT, Sofia University St. Kliment Ohridski", "ETH Zurich", "ELTE Eötvös Loránd University"]
description: "MIXAT 通过结合连续和离散对抗性攻击进行训练，显著提升了大型语言模型对多样化攻击的鲁棒性，同时保持了高效性和通用能力。"
---

> **Summary:** MIXAT 通过结合连续和离散对抗性攻击进行训练，显著提升了大型语言模型对多样化攻击的鲁棒性，同时保持了高效性和通用能力。 

> **Keywords:** LLM, Adversarial Training, Robustness, Continuous Attacks, Discrete Attacks

**Authors:** Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Institution(s):** INSAIT, Sofia University St. Kliment Ohridski, ETH Zurich, ELTE Eötvös Loránd University


## Problem Background

大型语言模型（LLMs）在面对对抗性攻击时仍然容易被诱导生成有害内容，尽管安全性和对齐研究已有进展。
现有对抗性训练方法存在局限：离散型攻击（如文本改写或后缀优化）虽有效但计算成本高，而连续型攻击（如嵌入空间扰动）虽高效却对多样化离散攻击缺乏鲁棒性。
论文旨在解决如何在保持计算效率的同时，提升 LLMs 对多种对抗性攻击的鲁棒性这一关键问题。

## Method

*   **核心思想**：提出 MIXAT（Mixed Adversarial Training），通过结合离散型和连续型对抗性攻击进行训练，扩展对抗性扰动空间，提升模型对未见攻击的泛化能力。
*   **具体实现**：
    *   **离散部分**：采用 PAP（一种低成本提示改写攻击）生成多样化的离散对抗性样本，作为训练的初始‘种子点’，覆盖文本层面的攻击形式。
    *   **连续部分**：基于 CAT 方法，在离散样本的嵌入空间上施加连续扰动（L2 球内优化，扰动幅度受控于参数 ϵ），以高效探索更广泛的对抗性空间。
    *   **混合策略**：引入混合参数 α，控制每个训练批次中连续攻击与离散攻击的比例（默认 α=0.5），确保模型同时学习两种攻击的特征。
    *   **训练目标**：使用综合损失函数，同时降低有害响应的概率、提升安全响应的概率，并通过效用数据集（如 UltraChat200k）加入效用损失以维持模型通用能力。
*   **关键优势**：不修改模型架构，仅通过训练策略调整实现鲁棒性提升，同时兼顾计算效率和多样化攻击防御。

## Experiment

*   **鲁棒性提升**：MIXAT 在多个模型（如 Zephyr-7B, Llama3-8B, Qwen2.5-14B/32B）上显著降低了对抗性攻击成功率，特别是在 ALO-ASR（At Least One Attack Success Rate）指标上，Zephyr-7B 模型的 ALO-ASR 降至 15%，远低于其他方法（如 R2D2, CAT, LAT）的 50% 以上，尤其对 jailbreak 类攻击（如 PAP, TAP, AutoDAN）表现出近乎完美的防御能力。
*   **效用维持**：在效用基准测试（如 ARC-E, ARC-C, MMLU）中，MIXAT 的性能下降幅度较小，例如在 Zephyr-7B 上 ARC-E 仅从 81.0% 降至 81.4%，优于其他防御方法，展现出良好的鲁棒性-效用权衡。
*   **计算效率**：训练成本低于 R2D2（16小时）和 CAT（6-7小时），MIXAT 在 Zephyr-7B 上仅需约 4小时（2xA100 GPU），略高于 LAT（1-2小时），整体效率较高。
*   **实验设置合理性**：实验覆盖多种模型规模和攻击类型（包括 PAP, GCG, TAP 等），并探讨了现实部署因素（如模型量化、LoRA 适配器缩放、生成温度影响），设置较为全面；但论文指出评估中的随机性和样本数量限制可能导致结果方差，需进一步优化。
*   **不足之处**：对 GCG 攻击的鲁棒性稍逊于专门针对 GCG 训练的 R2D2 方法，表明在特定 token 级攻击防御上仍有改进空间。

## Further Thoughts

MIXAT 的混合对抗性训练思路启发我们可以在其他领域（如图像或多模态模型）探索类似的多类型扰动组合策略，以提升全面鲁棒性；
动态生成对抗样本显著优于静态样本的实验结果提示，未来防御机制设计应注重实时适应性，或许可结合在线学习方法动态更新对抗样本；
此外，ALO-ASR 指标关注最坏情况下的模型脆弱性，这种视角对安全领域有重要意义，未来可设计更多综合性指标评估模型在复杂攻击场景下的表现。