---
title: "SCUBA: Salesforce Computer Use Benchmark"
pubDatetime: 2025-09-30T16:48:49+00:00
slug: "2025-09-scuba-crm-benchmark"
type: "arxiv"
id: "2509.26506"
score: 0.5511531904280179
author: "grok-3-latest"
authors: ["Yutong Dai", "Krithika Ramakrishnan", "Jing Gu", "Matthew Fernandez", "Yanqi Luo", "Viraj Prabhu", "Zhenyu Hu", "Silvio Savarese", "Caiming Xiong", "Zeyuan Chen", "Ran Xu"]
tags: ["VLM", "Computer Use Agent", "GUI Navigation", "Benchmark", "Enterprise Workflow"]
institution: ["Salesforce AI Research"]
description: "SCUBA 提出了一种真实的基准，用于评估计算机使用代理在 Salesforce 平台上的 CRM 工作流程性能，揭示了当前代理设计的局限性和演示增强的潜力。"
---

> **Summary:** SCUBA 提出了一种真实的基准，用于评估计算机使用代理在 Salesforce 平台上的 CRM 工作流程性能，揭示了当前代理设计的局限性和演示增强的潜力。 

> **Keywords:** VLM, Computer Use Agent, GUI Navigation, Benchmark, Enterprise Workflow

**Authors:** Yutong Dai, Krithika Ramakrishnan, Jing Gu, Matthew Fernandez, Yanqi Luo, Viraj Prabhu, Zhenyu Hu, Silvio Savarese, Caiming Xiong, Zeyuan Chen, Ran Xu

**Institution(s):** Salesforce AI Research


## Problem Background

随着大型视觉-语言模型（VLMs）的进步，自动化复杂企业工作流程（尤其是通过图形用户界面 GUI 完成的任务）成为研究热点，但现有基准（如 WebArena, OSWorld）无法充分捕捉企业软件环境的复杂性，尤其是在客户关系管理（CRM）领域。
论文指出，企业任务自动化不仅需要高成功率，还需考虑延迟和成本等实际部署因素，因此提出了 SCUBA 基准，旨在评估计算机使用代理在 Salesforce 平台上的表现，解决现有基准在真实企业场景中的不足。

## Method

*   **基准设计核心思想**：构建一个真实、全面的基准 SCUBA，用于评估计算机使用代理在 Salesforce 平台上的 CRM 工作流程表现，覆盖多角色任务和多维度指标。
*   **环境构建**：采用 Salesforce 沙盒环境，确保与生产环境一致的真实性；通过配置文件快照实现任务级环境重置，并支持并行评估以提高效率。
*   **任务设计**：基于真实用户访谈，构建 300 个任务实例，覆盖平台管理员、销售代表和服务代理三种角色，测试企业软件 UI 理解、信息检索、数据操作、工作流程构建和故障排除等能力；任务模板与 Salesforce Trailhead 知识文章配对，通过多样化查询生成增加难度和语言多样性。
*   **评估体系**：设计规则化评估器，提供任务成功与否的二元结果和细粒度的里程碑分数（过程奖励），以分析失败点；同时记录延迟（时间、步骤数）和成本（token 消耗、API 费用）等指标。
*   **辅助资源**：提供知识文章和人类演示数据，用于提升代理性能，尤其是在演示增强设置下。
*   **代理测试**：测试两类代理——浏览器使用代理（基于 SOM+DOM 文本观察空间）和计算机使用代理（基于屏幕截图观察空间），使用多种骨干模型（如 GPT-5, Claude-4-sonnet, UI-TARS 等）在零样本和演示增强设置下进行实验。

## Experiment

*   **有效性**：在零样本设置下，闭源模型（如 Agent-S2.5 with GPT-5）任务成功率最高达 39%，而开源模型（如 UI-TARS-1.5-7B）仅不到 5%，显示出显著性能差距；演示增强设置下，成功率普遍提升，最高达 50%（如 GPT-5），同时时间和成本分别降低 13% 和 16%。
*   **全面性**：实验覆盖多种代理设计范式（基础 VLM、原生 GUI 代理、代理框架）和多维度指标（成功率、里程碑分数、延迟、成本），设置合理且贴近企业需求；任务难度分布（易、中、难）和角色分布（管理员、销售、服务）设计均衡。
*   **局限性**：计算机使用代理在泛化能力和 grounding（定位 UI 元素）方面表现较差，尤其是在从 OSWorld 到 SCUBA 的迁移中性能下降明显；浏览器使用代理虽成功率较高，但依赖于对 Salesforce 平台的 DOM 解析器定制，通用性受限。
*   **总结**：演示增强是一种有效的改进策略，但代理在企业任务中的表现仍有较大提升空间，尤其是在 grounding 和泛化能力方面。

## Further Thoughts

SCUBA 基准的多维度评估（成功率、延迟、成本）为未来企业级代理设计提供了新思路，启发我们思考如何构建更贴近实际部署需求的评估体系；演示增强策略的成功提示未来可以探索如何利用非结构化文档或教程（而非依赖结构化演示数据）来提升代理性能；此外，计算机使用代理 grounding 问题的多重预测+多数投票解决方案，启发我们研究更鲁棒的视觉定位技术或混合观察空间（如结合 SOM 和屏幕截图）以提升代理在复杂 UI 环境中的表现。