---
title: "XBreaking: Explainable Artificial Intelligence for Jailbreaking LLMs"
pubDatetime: 2025-04-30T14:44:24+00:00
slug: "2025-04-xbreaking-jailbreaking"
type: "arxiv"
id: "2504.21700"
score: 0.4129162063621796
author: "grok-3-latest"
authors: ["Marco Arazzi", "Antonino Nocera", "Vignesh Kumar Kembu", "Vinod P."]
tags: ["LLM", "Explainable AI", "Jailbreaking", "Layer Analysis", "Noise Injection"]
institution: ["University of Pavia", "Cochin University of Science & Technology"]
description: "本文提出XBreaking方法，利用可解释性AI技术识别审查模型的关键层并注入噪声，成功绕过大型语言模型的安全限制，显著提升有害内容生成能力。"
---

> **Summary:** 本文提出XBreaking方法，利用可解释性AI技术识别审查模型的关键层并注入噪声，成功绕过大型语言模型的安全限制，显著提升有害内容生成能力。 

> **Keywords:** LLM, Explainable AI, Jailbreaking, Layer Analysis, Noise Injection

**Authors:** Marco Arazzi, Antonino Nocera, Vignesh Kumar Kembu, Vinod P.

**Institution(s):** University of Pavia, Cochin University of Science & Technology


## Problem Background

大型语言模型（LLMs）在AI领域中至关重要，但其安全性和隐私问题限制了在关键场景（如政府和医疗）中的应用。
为了防止生成有害内容，商业LLMs通常通过审查机制（如RLHF或外部分类器）进行内容过滤。
然而，LLM越狱（Jailbreaking）技术可以通过精心设计的输入绕过这些限制，生成违禁内容，现有方法多为生成-测试策略，缺乏对审查机制的深入理解。
本文旨在通过可解释性AI（XAI）分析审查与未审查模型的行为差异，设计一种针对性的越狱攻击方法。

## Method

*   **核心思想:** 利用可解释性AI（XAI）技术，分析审查模型（Mc）与未审查模型（Mu）的内部表征差异，识别负责内容审查的关键层，并通过有针对性的噪声注入绕过安全限制。
*   **具体步骤:**
    *   **内部表征分析:** 使用XAI技术，计算审查模型和未审查模型在各层的平均激活值（Activation）和注意力分数（Attention），并进行归一化处理，识别两模型间差异最大的层，揭示审查行为的内部机制。
    *   **关键层选择:** 将层级差异作为特征，构建二分类问题，通过特征选择技术（SelectKBest）识别对审查行为影响最大的层，利用肘部法则（Elbow Method）确定最优层数K，确保修改范围最小化。
    *   **噪声注入:** 在选定的关键层或其前一层注入高斯噪声，尝试两种策略：一是直接在目标层的自注意力查询矩阵（Q）中添加噪声，二是向前一层的层归一化权重中添加噪声，观察对安全限制的破坏效果，同时尽量保留模型原有功能。
*   **关键点:** 该方法基于白盒访问（即对模型内部结构完全可见），避免全面微调带来的广泛行为改变，注重精准定位和最小化干预，同时通过比较审查与未审查模型的行为差异，确保攻击的针对性。

## Experiment

*   **有效性:** 实验在四个开源LLM模型（LLaMA 3.2-1B/3B、Qwen2.5-3B、Mistral-7B-v0.3）上进行，使用JBB-Behaviors数据集（100个有害和良性提示），结果显示XBreaking方法显著提升了有害性评分（Harmfulness Score），尤其在前一层注入噪声时效果更佳（如LLaMA 3B和Qwen2.5分别提升60%和58%）。
*   **权衡性:** 噪声水平增加会导致响应相关性（Relevancy）下降，需在越狱效果与模型功能间权衡；Mistral因参数量大表现出较强鲁棒性，性能波动较小。
*   **合理性与局限:** 实验设置覆盖多种模型和噪声水平（0.1、0.2、0.3），采用‘LLM-as-a-Judge’评估方法，并通过人工标注验证（Cohen’s Kappa值为0.75，Judge LLM准确率80%），较为全面；但数据集规模较小，可能限制泛化性，且未探讨黑盒场景下的适用性。
*   **指纹化与层选择:** XAI分析成功识别审查与未审查模型的差异，指纹化准确率大多超80%（Mistral稍低）；关键层数量和位置因模型而异，LLaMA集中在后期层，Qwen2.5分布较广（19层）。

## Further Thoughts

XAI在安全研究中的潜力令人瞩目，不仅可用于越狱攻击，还能为改进安全机制提供思路，如精准强化审查层；此外，不同模型审查机制的分布差异（集中 vs. 广泛）提示架构和训练策略对安全性的深远影响，值得进一步探索；噪声注入的效果依赖层位置和强度，未来可研究自适应噪声策略，根据输入动态调整干扰，兼顾越狱效果与功能保留。