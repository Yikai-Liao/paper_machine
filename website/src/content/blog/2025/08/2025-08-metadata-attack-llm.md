---
title: "Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools"
pubDatetime: 2025-08-04T06:38:59+00:00
slug: "2025-08-metadata-attack-llm"
type: "arxiv"
id: "2508.02110"
score: 0.4550397205128925
author: "grok-3-latest"
authors: ["Kanghua Mo", "Li Hu", "Yucheng Long", "Zhihao Li"]
tags: ["LLM", "Tool Invocation", "Security Risk", "Metadata Optimization", "In-Context Learning"]
institution: ["School of Computer Science, Guangzhou University", "The Hong Kong Polytechnic University"]
description: "本文提出吸引性元数据攻击（AMA），通过优化工具元数据诱导 LLM 代理调用恶意工具，揭示了当前代理架构的系统性安全漏洞。"
---

> **Summary:** 本文提出吸引性元数据攻击（AMA），通过优化工具元数据诱导 LLM 代理调用恶意工具，揭示了当前代理架构的系统性安全漏洞。 

> **Keywords:** LLM, Tool Invocation, Security Risk, Metadata Optimization, In-Context Learning

**Authors:** Kanghua Mo, Li Hu, Yucheng Long, Zhihao Li

**Institution(s):** School of Computer Science, Guangzhou University, The Hong Kong Polytechnic University


## Problem Background

大型语言模型（LLM）代理通过调用外部工具来执行复杂任务（如金融分析、医疗决策），显著提升了自动化和泛化能力。然而，工具生态系统的开放性引入了新的安全风险：攻击者可以通过操纵工具元数据（如名称、描述、参数模式）来诱导代理优先选择恶意工具，而无需提示注入或访问模型内部。这种攻击利用了代理基于上下文和元数据进行工具选择的机制，构成了一种隐蔽且强大的威胁，可能会导致隐私泄露或任务误执行。

## Method

* **核心思想**：提出吸引性元数据攻击（Attractive Metadata Attack, AMA），通过系统性优化恶意工具的元数据，增加其被 LLM 代理选中的概率，从而实现隐蔽控制。
* **问题建模**：将元数据生成问题建模为状态-动作-价值（state-action-value）优化任务，其中状态是当前生成的恶意工具及其调用概率，动作是通过上下文学习生成新的工具元数据，价值函数评估工具的攻击潜力（即调用概率）。
* **优化机制**：引入三种约束机制以提高生成效率和效果：
  * **生成可追溯性**：记录每个工具的生成路径（父工具），以明确优化方向并加速收敛。
  * **加权价值评估**：综合考虑工具的绝对调用概率和相对于父工具的改进幅度，通过可调参数平衡两者重要性，筛选出最有潜力的工具。
  * **批量生成**：每次迭代为每个现有工具生成一批新工具，增加搜索的广度和多样性，提升优化效率。
* **迭代过程**：基于预收集的查询集和正常工具集，利用 LLM 的上下文学习能力，迭代生成和评估恶意工具元数据，直到调用概率达到预设阈值或迭代次数上限。
* **隐蔽性设计**：生成的元数据在语法和语义上合法，不干扰代理的执行框架，仅通过元数据吸引力影响工具选择行为。

## Experiment

* **有效性**：在10个现实工具使用场景和4个主流 LLM 代理（包括开源模型如 Gemma-3 27B、LLaMA-3.3 70B 和商业模型如 GPT-4o-mini）上，AMA 在目标攻击场景下攻击成功率（ASR）达到 81%-95%，隐私泄露率（PL）高达 92%-95%；在非目标攻击场景下也表现出较强泛化性。
* **隐蔽性**：任务完成率（TS）几乎不受影响，保持在 85%-99%，表明攻击不会干扰代理的主要任务执行，隐蔽性极高。
* **对比优势**：相比基线攻击（如提示攻击和注入攻击），AMA 在 ASR 和 PL 上均有显著提升（例如在开源模型上 ASR 提升 2%-19%），且与注入攻击结合时效果更佳。
* **防御失效**：现有提示级防御（如动态提示重写和提示庇护）对 AMA 几乎无效，甚至在某些情况下加剧攻击效果；结构化协议（如 Model Context Protocol）也仅对部分模型提供有限缓解。
* **实验设置合理性**：实验覆盖多种模型、场景和威胁模型（目标和非目标攻击），并评估了不同防御机制下的效果，设置较为全面；但未深入探讨工具生态系统差异对攻击效果的影响，可能存在一定局限性。

## Further Thoughts

AMA 揭示了工具元数据作为攻击面的重要性，提示我们在设计 LLM 代理时需关注工具选择机制的安全性，而不仅仅是提示或内容安全；其利用上下文学习进行攻击优化的思路可推广至其他对抗性场景，如生成对抗性输入；此外，元数据吸引力可能与 LLM 训练数据偏见相关，未来可研究通过调整训练数据降低此类偏见；最后，AMA 的成功表明提示级防御不足以应对复杂攻击，需在执行层面设计更强的安全机制，如工具验证或行为监控。