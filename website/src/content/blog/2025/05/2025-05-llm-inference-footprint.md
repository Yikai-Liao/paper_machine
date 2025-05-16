---
title: "How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference"
pubDatetime: 2025-05-14T17:47:00+00:00
slug: "2025-05-llm-inference-footprint"
type: "arxiv"
id: "2505.09598"
score: 0.4218298640883828
author: "grok-3-latest"
authors: ["Nidhal Jegham", "Marwan Abdelatti", "Lassad Elmoubarki", "Abdeltawab Hendawi"]
tags: ["LLM", "Energy Consumption", "Carbon Footprint", "Water Usage", "Infrastructure Efficiency", "Benchmarking", "Eco-Efficiency"]
institution: ["University of Rhode Island", "University of Tunis"]
description: "本文提出了一种基础设施感知框架，首次大规模量化了大型语言模型推理的每提示环境足迹，并通过跨效率分析揭示性能与可持续性的权衡，为 AI 部署的环保标准奠定基础。"
---

> **Summary:** 本文提出了一种基础设施感知框架，首次大规模量化了大型语言模型推理的每提示环境足迹，并通过跨效率分析揭示性能与可持续性的权衡，为 AI 部署的环保标准奠定基础。 

> **Keywords:** LLM, Energy Consumption, Carbon Footprint, Water Usage, Infrastructure Efficiency, Benchmarking, Eco-Efficiency

**Authors:** Nidhal Jegham, Marwan Abdelatti, Lassad Elmoubarki, Abdeltawab Hendawi

**Institution(s):** University of Rhode Island, University of Tunis


## Problem Background

大型语言模型（LLMs）在推理阶段的环境影响（能量、水和碳排放）已成为主要成本，占模型生命周期能耗的90%，但缺乏标准化方法来量化每提示（per-prompt）的环境足迹，尤其是在商业数据中心部署的专有模型上；现有框架无法处理专有模型、缺乏实时粒度或未能捕捉生产规模推理的基础设施复杂性，而 AI 提供商的数据不透明进一步阻碍了独立验证和政策制定。

## Method

* **核心思想**：提出一种基础设施感知的基准框架，通过整合公开数据和统计推断，量化 LLM 推理阶段的每提示环境足迹（能量、水和碳排放），并评估模型的生态效率。
* **具体实现**：
  * **数据整合**：从公共 API 获取模型性能指标（如延迟和每秒 token 数 TPS），结合已发布的 GPU 和系统功率规格，估算推理时间和能耗。
  * **硬件推断**：通过统计分析（如 ANOVA 和 Tukey HSD）推断30个模型的硬件配置（如 A100、H100、H200 GPU），并根据模型规模分配 GPU 数量和利用率。
  * **环境成本计算**：引入地区特定乘数，包括功率使用效率（PUE，考虑数据中心非计算开销）、水使用效率（WUE，计算现场冷却和电力生成的水耗）和碳强度因子（CIF，衡量每千瓦时碳排放），估算每查询的水耗和碳排放，聚焦运营阶段（Scope 2）。
  * **生态效率评估**：采用跨效率数据包络分析（DEA），以环境资源（能耗、水耗、碳排放）为输入，AI 性能指数（涵盖推理、数学和编码能力）为输出，评估模型如何将资源转化为功能智能，避免自我评估偏差。
* **关键创新**：方法综合考虑模型性能与基础设施因素，适用于开源和专有模型，填补了生产规模推理环境基准的空白。

## Experiment

* **有效性**：在30个模型上的测试显示，能耗差异显著，例如 o3 和 DeepSeek-R1 的长提示能耗超过33 Wh，是 GPT-4.1 nano 的70倍；水耗和碳排放也呈现类似模式，DeepSeek 模型因数据中心效率低而表现最差。
* **生态效率**：通过 DEA 评估，Claude-3.7 Sonnet 以高性能和低资源消耗排名生态效率最高，而 DeepSeek-R1 和 DeepSeek-V3 因高环境成本排名最低。
* **规模化影响**：GPT-4o 案例研究表明，单次短提示能耗仅0.42 Wh，但按2025年每日7亿查询估算，年能耗达391,509-463,269 MWh，相当于35,000个美国家庭用电，水耗相当于1.2百万人的年饮水需求，碳排放需芝加哥大小的森林抵消。
* **合理性与局限**：实验设置全面，涵盖多种模型、提示长度和地区差异，但对硬件利用率和批大小的假设（如批大小为8）可能影响精度；排除 Scope 3 排放可能低估全生命周期影响。

## Further Thoughts

论文揭示了基础设施在 AI 可持续性中的关键作用，启发我们未来应优先优化部署环境（如采用更高效的硬件和可再生能源）而不仅是模型算法；Jevons Paradox 的应用提示效率提升可能加剧总资源消耗，建议探索系统性约束机制，如通过政策设定推理环境足迹阈值；此外，跨效率 DEA 的方法可扩展到其他 AI 领域（如图像生成），以平衡性能与环境成本。