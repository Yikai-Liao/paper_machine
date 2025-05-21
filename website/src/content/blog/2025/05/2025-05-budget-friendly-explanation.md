---
title: "Towards Budget-Friendly Model-Agnostic Explanation Generation for Large Language Models"
pubDatetime: 2025-05-18T18:05:37+00:00
slug: "2025-05-budget-friendly-explanation"
type: "arxiv"
id: "2505.12509"
score: 0.818097270885706
author: "grok-3-latest"
authors: ["Junhao Liu", "Haonan Yu", "Xin Zhang"]
tags: ["LLM", "Proxy Model", "Explanation Generation", "Cost Reduction", "Model-Agnostic"]
institution: ["Key Lab of High Confidence Software Technologies (Peking University)", "School of Computer Science, Peking University"]
description: "本文提出了一种利用预算友好型模型生成代理解释的方法，显著降低了大型语言模型解释生成的经济成本，同时保持了高保真度和下游任务表现。"
---

> **Summary:** 本文提出了一种利用预算友好型模型生成代理解释的方法，显著降低了大型语言模型解释生成的经济成本，同时保持了高保真度和下游任务表现。 

> **Keywords:** LLM, Proxy Model, Explanation Generation, Cost Reduction, Model-Agnostic

**Authors:** Junhao Liu, Haonan Yu, Xin Zhang

**Institution(s):** Key Lab of High Confidence Software Technologies (Peking University), School of Computer Science, Peking University


## Problem Background

随着大型语言模型（LLMs）在各种应用中的普及，解释其预测结果对终端用户的需求日益增加。
由于LLM架构多样且部分为闭源，模型不可知解释技术因无需访问模型内部参数而备受关注。
然而，现有方法（如LIME和SHAP）需要多次调用目标模型生成解释，导致经济成本高昂，尤其对于商业模型如GPT-4o。
论文解决的关键问题是：如何在不牺牲解释质量的情况下，显著降低生成解释的经济成本？

## Method

*   **核心思想:** 利用预算友好型模型（budget-friendly models，通常为开源或低成本模型）生成代理解释（proxy explanations），作为昂贵模型解释的替代，以降低成本。
*   **具体实现步骤:**
    *   选择预算友好型模型（如Qwen 2.5系列、LLaMA 3.1），这些模型通常参数规模较小或可本地运行，调用成本低。
    *   在这些模型上应用模型不可知解释技术（如LIME和Kernel SHAP），通过扰动输入生成大量样本，构建局部解释，描述模型在特定输入附近的决策行为。
    *   将预算友好型模型生成的解释作为代理解释，应用于昂贵模型（如GPT-4o），并评估其保真度（即解释是否准确反映目标模型行为）和下游任务表现。
    *   引入优化策略，如在问答任务中过滤掉预算友好型模型与昂贵模型预测不一致的样本，以提升代理解释的保真度。
*   **关键优势:** 不需要多次调用昂贵模型，仅通过低成本模型生成解释即可近似反映昂贵模型的行为，显著降低经济开销，同时保持解释的保真度和实用性。
*   **适用范围:** 方法适用于多种任务，包括情感分析、问答和文本生成，且对不同规模和架构的模型均有效。

## Experiment

*   **有效性:** 实验表明，代理解释在保真度上表现优异。例如，在情感分析任务中，Qwen 2.5-7B生成的LIME解释对GPT-4o的保真度高达97%，成本仅为本地运行电费；在问答任务中，过滤预测不一致样本后，Qwen 2.5-0.5B的保真度提升至83%；文本生成任务中，所有模型的代理解释保真度与原解释相当。
*   **下游任务表现:** 在上下文学习（ICL）的提示压缩任务中，使用小模型（如Qwen 2.5-7B）生成的解释进行提示删减，移除比例在83.33%至95.83%之间，远高于随机删减的58.97%至83.93%，表明代理解释在实际应用中有效。
*   **实验设置合理性:** 实验覆盖12个模型（参数规模从0.5B到685B），包括开源和闭源模型，任务涵盖情感分析（SST-2）、问答（MMLU）和文本生成（NQ），指标包括accuracy和AOPC，设计较为全面。但模型选择标准未明确，可能存在主观性，且未探讨极端情况下代理解释的失效场景。
*   **成本效益:** 相比直接调用昂贵模型生成解释，代理解释方法将成本降低至接近于零（对于免费API或本地模型），效果提升明显。

## Further Thoughts

论文中代理解释的跨模型适用性启发了我思考：是否可以通过‘解释蒸馏’机制，针对特定任务或领域训练一个专门的低成本模型来生成解释？此外，过滤策略（排除预测不一致样本）提示了一种混合策略：是否可以在运行时动态决定是否使用代理解释，例如根据输入特性或模型预测置信度，结合昂贵模型的少量调用，进一步提升保真度和鲁棒性？