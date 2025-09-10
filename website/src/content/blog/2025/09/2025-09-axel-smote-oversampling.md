---
title: "AxelSMOTE: An Agent-Based Oversampling Algorithm for Imbalanced Classification"
pubDatetime: 2025-09-08T16:47:33+00:00
slug: "2025-09-axel-smote-oversampling"
type: "arxiv"
id: "2509.06875"
score: 0.5958897298745274
author: "grok-3-latest"
authors: ["Sukumar Kishanthan", "Asela Hevapathige"]
tags: ["Class Imbalance", "Oversampling", "Agent-Based Modeling", "Synthetic Data", "Feature Correlation"]
institution: ["Dialog Axiata PLC", "Australian National University"]
description: "本文提出 AxelSMOTE，一种基于代理的过采样方法，通过模拟数据实例间的文化传播交互生成真实、多样化的合成样本，显著提升不平衡分类任务性能。"
---

> **Summary:** 本文提出 AxelSMOTE，一种基于代理的过采样方法，通过模拟数据实例间的文化传播交互生成真实、多样化的合成样本，显著提升不平衡分类任务性能。 

> **Keywords:** Class Imbalance, Oversampling, Agent-Based Modeling, Synthetic Data, Feature Correlation

**Authors:** Sukumar Kishanthan, Asela Hevapathige

**Institution(s):** Dialog Axiata PLC, Australian National University


## Problem Background

类别不平衡问题是机器学习中的一大挑战，传统过采样方法（如 SMOTE）在生成合成样本时存在多个缺陷：忽略特征间相关性、缺乏基于相似性的生成控制、生成过程过于确定性导致样本多样性不足，以及缺乏对合成样本多样性的有效管理。
AxelSMOTE 旨在通过引入基于代理的视角，将数据实例视为自主代理，模拟复杂交互，生成更真实、多样化的少数类样本，从而提升分类器在不平衡数据集上的性能。

## Method

*   **核心思想:** 将数据实例视为自主代理（Agents），基于 Axelrod 文化传播模型，通过代理间的交互生成合成样本，解决传统过采样方法的局限性。
*   **具体实现:** 
    *   **特征特质分组（Trait-Based Feature Grouping）**：将特征划分为多个特质组，确保语义相关的特征在生成过程中被集体修改，保留特征间相关性。
    *   **基于相似性的概率交换机制（Similarity-Based Probabilistic Exchange）**：在生成合成样本时，随机选择一个基础样本及其 k 个最近邻，仅当特质相似性超过阈值（θ）且满足概率条件（α）时，执行特征交换，避免生成不合理样本。
    *   **Beta 分布混合（Beta Distribution Blending）**：在特征交换时，从 Beta(2,2) 分布中采样混合比例，实现自然插值，避免极端值，确保合成样本的平滑性。
    *   **受控多样性注入（Controlled Diversity Injection）**：以概率 α 为交换后的特质添加高斯噪声，基于特征范围进行缩放，增强样本多样性，防止过拟合。
*   **关键优势:** 通过代理交互模拟数据生成过程，既保留了数据内在结构，又通过概率机制和噪声注入增加多样性，同时不依赖特定分类器，具有模型无关性。

## Experiment

*   **有效性:** AxelSMOTE 在八个不平衡数据集上的 F1 分数和平衡准确率均优于基线方法，平均 F1 分数比 SMOTE 提高 2.37%，在多个数据集（如 Page Blocks、Thyroid）上表现尤为突出，表明其生成的合成样本显著提升了分类器对少数类的识别能力。
*   **实验设置合理性:** 实验采用 80:20 训练-测试划分，10 次独立运行确保统计稳健性，数据集涵盖二分类和多分类任务，超参数通过网格搜索优化，与基线方法公平比较。
*   **组件贡献:** 消融研究表明每个组件（如 Beta 分布混合、相似性过滤）均对性能有贡献，其中 Beta 分布混合影响最大，表明其对插值质量的关键作用。
*   **计算效率:** 运行时间略高于基本 SMOTE 变体，但优于复杂方法（如 SMOTENC），在性能与效率间取得平衡。
*   **样本质量:** t-SNE 可视化显示 AxelSMOTE 生成的样本在类间分离和类内聚类上表现最佳，接近真实数据分布。

## Further Thoughts

AxelSMOTE 引入社会学模型（如 Axelrod 文化传播模型）来解决数据生成问题，这种跨学科思维启发我们是否可以借鉴其他社会或生物学模型（如群体智能、进化算法）设计过采样方法，通过模拟‘竞争与合作’机制进一步提升样本多样性；此外，是否可以通过自适应学习（如强化学习）动态调整超参数，以适应不同数据集特性，或将代理交互机制扩展到时间序列、图像等复杂数据类型，通过定义特定‘特质’捕捉数据依赖关系？