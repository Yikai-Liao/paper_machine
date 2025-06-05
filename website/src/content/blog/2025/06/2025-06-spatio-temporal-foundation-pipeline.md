---
title: "Unraveling Spatio-Temporal Foundation Models via the Pipeline Lens: A Comprehensive Review"
pubDatetime: 2025-06-02T06:46:42+00:00
slug: "2025-06-spatio-temporal-foundation-pipeline"
type: "arxiv"
id: "2506.01364"
score: 0.5721447128284731
author: "grok-3-latest"
authors: ["Yuchen Fang", "Hao Miao", "Yuxuan Liang", "Liwei Deng", "Yue Cui", "Ximu Zeng", "Yuyang Xia", "Yan Zhao", "Torben Bach Pedersen", "Christian S. Jensen", "Xiaofang Zhou", "Kai Zheng"]
tags: ["Spatio-Temporal Data", "Foundation Model", "Data Harmonization", "Model Taxonomy", "Self-Supervised Learning"]
institution: ["University of Electronic Science and Technology of China", "Aalborg University", "Hong Kong University of Science and Technology", "Hong Kong University of Science and Technology (Guangzhou)"]
description: "本文通过管道视角系统综述时空基础模型（STFMs），提出数据属性分类法，梳理从数据协调到应用的完整流程，并指明未来研究方向，为研究者提供全面框架。"
---

> **Summary:** 本文通过管道视角系统综述时空基础模型（STFMs），提出数据属性分类法，梳理从数据协调到应用的完整流程，并指明未来研究方向，为研究者提供全面框架。 

> **Keywords:** Spatio-Temporal Data, Foundation Model, Data Harmonization, Model Taxonomy, Self-Supervised Learning

**Authors:** Yuchen Fang, Hao Miao, Yuxuan Liang, Liwei Deng, Yue Cui, Ximu Zeng, Yuyang Xia, Yan Zhao, Torben Bach Pedersen, Christian S. Jensen, Xiaofang Zhou, Kai Zheng

**Institution(s):** University of Electronic Science and Technology of China, Aalborg University, Hong Kong University of Science and Technology, Hong Kong University of Science and Technology (Guangzhou)


## Problem Background

时空数据在交通、天气、能源等领域广泛存在，但传统深度学习模型多为任务特定的‘一对一’设计，每个任务需单独训练，导致计算和存储成本高昂。
随着自监督学习和规模法则的发展，‘一对多’的时空基础模型（STFMs）应运而生，旨在通过单一通用模型解决多种时空任务，降低资源消耗。
然而，现有综述缺乏对 STFMs 设计、训练和适配全流程的系统性分析，数据与模型联系薄弱，忽略数据属性对模型选择的影响，研究呈现碎片化。

## Method

*   **管道视角（Pipeline Lens）**：论文提出一个系统性框架，将 STFMs 的研究分为四个阶段：
    *   **数据协调（Data Harmonization）**：包括数据预处理（如噪声过滤、特征提取）、嵌入技术（如空间嵌入、时间嵌入、频率嵌入）以及侧信息整合（如外部文本、地理特征），以对齐原始时空数据与深度学习模型输入。
    *   **模型构建（Model Construction）**：基于数据属性分类法（Data Property Taxonomy），将 STFMs 分为原始模型（直接在时空数据上训练）和迁移模型（从语言或视觉领域迁移知识）；原始模型按时间、空间、时空依赖性细分，迁移模型按视觉、语言、多模态细分，为模型设计和选择提供指导。
    *   **训练与适配（Training and Adaptation）**：原始模型采用回归建模、掩码建模、对比学习和扩散生成等自监督训练目标，学习通用时空知识；迁移模型通过提示工程、特征增强、跨域对齐和监督微调等技术，将预训练模型适配到时空任务。
    *   **应用（Application）**：展示 STFMs 在能源、金融、天气、医疗、交通和公共服务等领域的广泛应用，验证其‘一对多’的通用性。
*   **数据属性分类法**：创新性地从数据来源和依赖性角度对 STFMs 进行分类，避免了传统基于数据类型或方法的粗粒度分类，提供更细致的模型设计依据。
*   **未来方向探讨**：提出可扩展性、效率、泛化能力、基准测试、多目标训练和多模态模型等研究机会，为领域发展提供指引。

## Experiment

*   **综述性质限制**：作为综述性论文，未提供具体实验数据或模型性能对比，而是通过总结现有研究和数据集（如表 II 列出的轨迹、事件、时空网格、视频、时空图数据集）评估 STFMs 的应用效果。
*   **应用效果**：STFMs 在天气预测、交通优化等领域展现出显著优势，尤其在减少任务特定模型训练成本方面表现突出，例如在智能交通系统中优化信号灯控制、降低拥堵。
*   **面临挑战**：文中指出 STFMs 面临数据规模较小（相比大型语言模型的训练数据）、模型复杂性高、计算成本大、泛化能力不足等问题；数据集异质性（如空间尺度、时间范围、采样间隔差异）增加了训练和部署难度。
*   **全面性对比**：通过表 I 对比其他相关综述，本文覆盖了更广泛的数据类型、训练目标和适配技术，体现了研究的全面性和系统性。

## Further Thoughts

论文提出的‘数据属性分类法’启发我们可以在其他数据驱动领域尝试类似的属性驱动模型设计，例如在多模态数据中基于模态依赖性进行分类，以优化模型选择；‘多目标训练’方向也具有潜力，是否可以通过动态权重调整机制，根据任务需求自适应平衡回归、掩码建模等目标的贡献？此外，多模态 STFMs 的发展方向值得探索，结合时空数据与文本、图像等多模态输入，通过跨模态对齐是否能显著提升模型对缺失或噪声数据的鲁棒性？