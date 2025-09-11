---
title: "Disentangling Interaction and Bias Effects in Opinion Dynamics of Large Language Models"
pubDatetime: 2025-09-08T16:26:45+00:00
slug: "2025-09-llm-opinion-dynamics"
type: "arxiv"
id: "2509.06858"
score: 0.6475027433829486
author: "grok-3-latest"
authors: ["Vincent C. Brockers", "David A. Ehrlich", "Viola Priesemann"]
tags: ["LLM", "Opinion Dynamics", "Bias Effects", "Interaction Modeling", "Bayesian Framework"]
institution: ["Max-Planck-Institute for Dynamics and Self-Organization, Göttingen, Germany", "Institute for the Dynamics of Complex Systems, University of Göttingen, Göttingen, Germany", "Campus Institute for Dynamics of Biological Networks, University of Göttingen, Göttingen, Germany"]
description: "本文提出一种贝叶斯框架，成功分离并量化了大型语言模型在意见动态中的交互效应和偏见效应，为模型行为的可解释性和跨模型比较提供了有力工具。"
---

> **Summary:** 本文提出一种贝叶斯框架，成功分离并量化了大型语言模型在意见动态中的交互效应和偏见效应，为模型行为的可解释性和跨模型比较提供了有力工具。 

> **Keywords:** LLM, Opinion Dynamics, Bias Effects, Interaction Modeling, Bayesian Framework

**Authors:** Vincent C. Brockers, David A. Ehrlich, Viola Priesemann

**Institution(s):** Max-Planck-Institute for Dynamics and Self-Organization, Göttingen, Germany, Institute for the Dynamics of Complex Systems, University of Göttingen, Göttingen, Germany, Campus Institute for Dynamics of Biological Networks, University of Göttingen, Göttingen, Germany


## Problem Background

大型语言模型（LLMs）在模拟人类意见动态时表现出强大潜力，但其行为受到训练数据中的固有偏见（包括主题偏见、同意偏见和锚定偏见）的影响，掩盖了真正的交互效应，导致模拟结果与人类行为存在偏差。
研究的关键问题是开发一个系统性框架，分离并量化交互作用与偏见的影响，以提高 LLMs 作为人类行为代理的可靠性和可解释性。

## Method

*   **模拟讨论设置**：通过两个 LLM 代理进行多轮对话，针对特定主题（如气候变化、AI 安全、财富分配）初始化不同意见，使用五点量表（-2 到 +2）测量意见变化，模拟 25 次以确保统计可靠性。
*   **贝叶斯建模框架**：构建一个影响-响应函数，将意见变化建模为交互效应（基于代理间意见差异，包含时间衰减项）和三种偏见效应（主题偏见、同意偏见、锚定偏见）的叠加，通过贝叶斯推理估计各因素的参数和效应大小。
*   **意见量化方式**：采用二维表示法，意见期望值表示立场，香农熵表示不确定性，熵值用于预测后续意见变化的方差。
*   **微调策略**：对 Mixtral-8x7B 模型进行参数高效微调（使用 LoRA 方法），基于气候变化主题的意见标注数据，测试是否能增强初始意见的坚持度和交互效应。
*   **模型比较**：在三种 LLM（DolphinMixtral、Mixtral-8x7B、GPT-4o-mini）上应用上述方法，分析不同模型中交互与偏见的相对重要性。

## Experiment

*   **偏见与交互差异**：实验结果表明，不同 LLM 受偏见影响程度不同，DolphinMixtral 主要受同意偏见驱动（效应大小最高），Mixtral-8x7B 受主题偏见主导，而 GPT-4o-mini 表现出更强的交互效应（交互强度 α 较高），偏见影响较小。
*   **意见收敛性**：所有模型的意见轨迹快速收敛到共享吸引子，交互效应在初期显著，但随时间快速衰减（时间尺度 τ 约为 0.3-0.5 轮），表明 LLMs 难以模拟人类意见的长期顽固性。
*   **微调效果**：微调后的 Mixtral 模型在初始意见坚持度上有所提升（主题吸引子与初始意见相关性更强），交互效应略有增强（τ 略高），但主题偏见仍占主导。
*   **不确定性预测**：香农熵作为不确定性指标，与后续意见变化方差显著相关（相关系数 r=0.88），验证了其有效性。
*   **实验设置评价**：实验覆盖多个主题和初始意见组合，统计可靠性较高，但仅限于三个主题和两代理对话，泛化性受限，且未涉及更复杂的多代理网络或长期记忆效应。

## Further Thoughts

贝叶斯框架不仅限于意见动态，还可扩展至其他社会行为模拟（如合作博弈或道德决策），为 LLM 内部过程提供可解释性工具；香农熵作为不确定性指标的思路可应用于评估模型在生成任务中的信心；微调调整吸引子位置的策略启发我们可以通过定制化训练数据控制 LLM 行为倾向，可能用于减少有害偏见或增强特定交互模式。