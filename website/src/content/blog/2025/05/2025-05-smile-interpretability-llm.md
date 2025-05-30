---
title: "Explainability of Large Language Models using SMILE: Statistical Model-agnostic Interpretability with Local Explanations"
pubDatetime: 2025-05-27T18:32:38+00:00
slug: "2025-05-smile-interpretability-llm"
type: "arxiv"
id: "2505.21657"
score: 0.631100248131046
author: "grok-3-latest"
authors: ["Zeinab Dehghani", "Koorosh Aslansefat", "Adil Khan", "Mohammed Naveed Akram"]
tags: ["LLM", "Explainable AI", "Interpretability", "Perturbation", "Visualization"]
institution: ["University of Hull", "Fraunhofer IESE"]
description: "本文提出 SMILE 框架，通过输入扰动和 Wasserstein 距离分析，生成可视化热图揭示大型语言模型对输入提示的响应机制，从而提升模型透明度和用户信任。"
---

> **Summary:** 本文提出 SMILE 框架，通过输入扰动和 Wasserstein 距离分析，生成可视化热图揭示大型语言模型对输入提示的响应机制，从而提升模型透明度和用户信任。 

> **Keywords:** LLM, Explainable AI, Interpretability, Perturbation, Visualization

**Authors:** Zeinab Dehghani, Koorosh Aslansefat, Adil Khan, Mohammed Naveed Akram

**Institution(s):** University of Hull, Fraunhofer IESE


## Problem Background

大型语言模型（LLMs）如 GPT、LLaMA 和 Claude 在文本生成等方面表现出色，但其‘黑箱’特性导致缺乏透明度和可解释性，尤其在需要信任和问责的领域（如医疗、教育、法律）中，这是一个重大问题。
论文旨在解决 LLMs 的可解释性问题，聚焦于揭示模型对输入提示（prompt）中不同部分的响应机制，以提升用户对模型决策过程的理解和信任。

## Method

*   **核心思想:** 提出 SMILE（Statistical Model-agnostic Interpretability with Local Explanations），一种基于统计的模型无关的可解释性方法，通过分析输入扰动对输出的影响，揭示 LLMs 中哪些输入词或短语对生成结果影响最大。
*   **具体实现步骤:** 
    *   **输入扰动生成:** 将原始输入提示分解为单个词，通过选择性包含或排除某些词生成多个扰动提示（perturbed prompts）。
    *   **语义与输出变化量化:** 使用 Word Mover’s Distance (WMD) 计算扰动提示与原始提示的语义距离，并使用 Wasserstein 距离量化模型输出分布的变化，以捕捉输入变化对输出的影响。
    *   **加权机制:** 基于语义距离，通过高斯核函数为每个扰动分配权重，语义上更接近原始输入的扰动获得更高权重。
    *   **局部代理模型:** 使用加权线性回归模型拟合扰动输入与输出变化之间的关系，回归系数反映每个输入词的重要性；此外，还测试了 Bayesian Ridge 作为替代代理模型。
    *   **可视化输出:** 生成热图（heatmap），直观展示输入提示中每个词对输出的影响程度，颜色深浅表示重要性高低。
*   **关键特点:** 方法不依赖特定模型结构，具有通用性；通过统计距离提升解释的鲁棒性，但计算复杂度较高。

## Experiment

*   **有效性:** SMILE 在多个 LLMs（如 OpenAI GPT、LLaMA、Claude-AI）上测试，生成的热图清晰展示了输入词的重要性，例如在 MMLU 数据集上成功识别关键问题词对输出的影响；定量结果显示 Claude-AI 在准确性（ATT AUROC 0.88）和保真度（R²ω 0.7209）上表现最佳。
*   **全面性与合理性:** 实验设置涵盖不同模型、扰动数量（32到256）、距离度量（Wasserstein 距离与 Cosine 相似度）和代理模型（加权线性回归与 Bayesian Ridge），评估指标包括准确性、稳定性、一致性和保真度，较为全面；此外，使用了多种场景（如不同句式、领域特定术语）测试方法的适应性。
*   **局限性:** 稳定性指标（如 LLaMA 的 Jaccard Index 仅为 0.45）显示解释对输入微小变化的敏感性较高；计算复杂度是明显短板，SMILE 使用 Wasserstein 距离导致执行时间较长（如在 OpenAI GPT 上为 170.70 秒，远高于 LIME 的 156.82 秒）。
*   **权衡分析:** 随着扰动数量增加，误差（如 WMSE）通常降低，但对保真度（R²ω）的提升有限，显示计算资源与解释质量之间的权衡。

## Further Thoughts

SMILE 的输入扰动与输出变化映射方法启发我们，不仅可以用于文本生成任务，还可能扩展到多模态任务（如图像-文本生成）中，通过分析跨模态输入的影响提升模型透明度；此外，强调输入空间可解释性的思路为提示工程（prompt engineering）提供了新视角，用户可通过优化关键输入词提升模型表现；最后，统计距离（如 Wasserstein 距离）在捕捉语义变化方面的潜力值得进一步探索，未来可以结合近似算法或混合距离度量以平衡精度和效率。