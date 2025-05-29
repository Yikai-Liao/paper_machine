---
title: "Be Decisive: Noise-Induced Layouts for Multi-Subject Generation"
pubDatetime: 2025-05-27T17:54:24+00:00
slug: "2025-05-noise-induced-layouts"
type: "arxiv"
id: "2505.21488"
score: 0.5080398752099827
author: "grok-3-latest"
authors: ["Omer Dahary", "Yehonathan Cohen", "Or Patashnik", "Kfir Aberman", "Daniel Cohen-Or"]
tags: ["Diffusion Model", "Multi-Subject Generation", "Noise Prior", "Layout Prediction", "Attention Control"]
institution: ["Tel Aviv University", "Snap Research"]
description: "本文提出基于初始噪声的布局预测与动态精炼方法，解决了文本到图像扩散模型在多主体生成中的主体泄露问题，同时保持模型先验分布的多样性和自然性。"
---

> **Summary:** 本文提出基于初始噪声的布局预测与动态精炼方法，解决了文本到图像扩散模型在多主体生成中的主体泄露问题，同时保持模型先验分布的多样性和自然性。 

> **Keywords:** Diffusion Model, Multi-Subject Generation, Noise Prior, Layout Prediction, Attention Control

**Authors:** Omer Dahary, Yehonathan Cohen, Or Patashnik, Kfir Aberman, Daniel Cohen-Or

**Institution(s):** Tel Aviv University, Snap Research


## Problem Background

文本到图像扩散模型在生成多主体图像时面临主体泄露（Subject Leakage）问题，导致生成的图像在主体数量、属性或视觉特征上与复杂提示不符。
以往方法依赖外部预定义布局来控制主体位置，但这种布局往往与初始噪声隐含的布局冲突，偏离模型先验分布，影响图像质量和语义对齐。
本文旨在利用初始噪声中的布局信息，动态生成与提示对齐的空间布局，避免外部布局冲突，提升多主体生成的准确性和多样性。

## Method

*   **核心思想：** 从初始噪声中提取布局信息，并在去噪过程中动态精炼布局，指导多主体图像生成，避免外部布局与模型先验的冲突。
*   **软布局预测：** 训练一个小型神经网络，利用去噪模型的注意力层特征，预测软布局（Soft-Layout），即每个像素与其他像素关联为同一主体的概率特征图，初期粗糙，后期逐渐精细。
*   **硬布局生成：** 通过 K-Means 聚类将软布局转化为硬布局（Hard-Layout），即明确的主体区域分割，并结合提示中的主体数量和交叉注意力图，使用匈牙利算法为每个区域分配主体标签。
*   **果断引导机制：** 在每个去噪步骤后，通过优化潜在图像，使预测的软布局与前一步硬布局对齐，优化目标包括交叉注意力损失（确保语义对齐）、方差损失（减少簇内歧义）和 Dice 损失（保持时间步间边界一致性）。
*   **边界注意力控制：** 采用已有方法，通过掩码限制不同主体间的注意力交互，减少主体泄露，确保各主体视觉特征的独立性。
*   **关键创新：** 不依赖外部布局，基于噪声先验动态生成并精炼布局，保持模型分布的自然性，同时通过引导机制确保主体边界清晰。

## Experiment

*   **有效性：** 定性结果显示，该方法在不同随机种子下生成的图像布局多样，准确反映提示中的主体类别、属性和数量，明显优于基线方法（如 SDXL、Attend-and-Excite、LLM+BA）在避免主体泄露和生成自然构图方面的表现。
*   **定量提升：** 在 T2I-CompBench 数据集上，该方法在多类别组合（0.723）、属性绑定（0.686）、数量准确性（0.837）和布局多样性（0.718）等指标上均表现优异，尤其在布局多样性上显著高于 LLM 布局方法（0.718 vs. 0.408）。
*   **用户研究：** 在复杂多主体提示下，用户更倾向选择本文方法生成的图像（例如 vs. SDXL 概率为 0.74，vs. LLM+BA 为 0.87），验证了其优越性。
*   **实验设置：** 实验全面，涵盖单类别、多类别及个性化主体提示，与多种基线方法对比，消融研究验证了各组件贡献；但计算成本较高（采样时间约77秒，SDXL 为7秒），且在拥挤场景或复杂空间关系提示下仍有不足。
*   **总结：** 方法提升显著，尤其在准确性和多样性平衡上表现突出，实验设计合理，但计算开销和特定场景局限性需进一步优化。

## Further Thoughts

初始噪声中隐含的布局信息作为天然引导信号的潜力，启发我们在其他生成任务中探索噪声先验特性，而非仅依赖外部条件；
动态精炼布局的‘果断决策’思路，可应用于视频生成或3D建模等需要逐步决策的任务；
通过边界注意力限制主体间干扰的机制，提示在多模态任务中可类似控制不同元素间的交互，提升生成一致性。