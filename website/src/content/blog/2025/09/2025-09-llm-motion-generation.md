---
title: "Do LLM Modules Generalize? A Study on Motion Generation for Autonomous Driving"
pubDatetime: 2025-09-02T19:02:49+00:00
slug: "2025-09-llm-motion-generation"
type: "arxiv"
id: "2509.02754"
score: 0.4926348674606656
author: "grok-3-latest"
authors: ["Mingyi Wang", "Jingke Wang", "Tengju Ye", "Junbo Chen", "Kaicheng Yu"]
tags: ["LLM", "Autonomous Driving", "Motion Generation", "Tokenization", "Positional Encoding", "Pre-Training", "Post-Training", "Test Time Computing"]
institution: ["Westlake University", "UDEER.AI", "Zhejiang University"]
description: "本文首次系统研究了大型语言模型（LLM）模块在自动驾驶运动生成中的可迁移性，通过针对性适配五个核心模块显著提升性能，并在Waymo Sim Agents基准上取得竞争性结果。"
---

> **Summary:** 本文首次系统研究了大型语言模型（LLM）模块在自动驾驶运动生成中的可迁移性，通过针对性适配五个核心模块显著提升性能，并在Waymo Sim Agents基准上取得竞争性结果。 

> **Keywords:** LLM, Autonomous Driving, Motion Generation, Tokenization, Positional Encoding, Pre-Training, Post-Training, Test Time Computing

**Authors:** Mingyi Wang, Jingke Wang, Tengju Ye, Junbo Chen, Kaicheng Yu

**Institution(s):** Westlake University, UDEER.AI, Zhejiang University


## Problem Background

大型语言模型（LLM）在自然语言处理领域的成功启发了其模块化技术向自动驾驶运动生成任务的迁移。
自动驾驶中的运动生成（如轨迹预测、交通仿真、自车规划）与语言生成在自回归序列建模、token表示和上下文感知决策等方面有相似性，但目前缺乏对LLM模块可迁移性的系统性研究。
论文试图解决的问题是：哪些LLM模块可以直接应用于自动驾驶运动生成，哪些需要领域特定适配，以及如何进行适配。

## Method

* **Tokenizing（分词）**：针对运动轨迹的连续性特点，提出基于模型的Verlet-Agent方法，将轨迹在代理中心坐标系中编码为离散token（词汇量169），通过一致性映射提升模型学习效率，优于数据驱动方法和全局坐标系编码。
* **Positional Embedding（位置编码）**：提出Global-DRoPE方法，将代理和地图信息编码在全局坐标系中，同时在注意力机制中引入相对位置线索（基于DRoPE框架），以保留丰富的空间语义信息，增强模型对复杂拓扑结构的空间推理能力。
* **Pre-training（预训练）**：采用自回归下一token预测范式，在Waymo Open Motion Dataset上从头训练Transformer解码器模型，设计包括时间自注意力、代理间自注意力、地图交叉注意力等模块，并通过数据增强策略增加训练数据多样性，验证了数据量和模型参数规模的scaling law。
* **Post-training（后训练）**：测试多种后训练策略，包括监督微调（SFT）针对安全关键场景微调，以及强化学习方法如REINFORCE、A2C和GRPO，通过环境反馈优化轨迹安全性，其中GRPO通过组内优势比较和KL散度正则化实现安全性和人类行为相似性的最佳平衡。
* **Test-time Computing（测试时计算）**：在推理阶段通过并行生成多条轨迹（rollouts），结合安全过滤搜索和K-Medoids聚类策略，从中选择最安全和多样化的轨迹输出，以提升规划质量。

## Experiment

* **有效性**：在Waymo Sim Agents基准上，各模块适配后显著提升性能，例如Verlet-Agent方法在ADE（平均位移误差）上从3.53降至3.13，Global-DRoPE降低碰撞率和越界率，GRPO后训练策略在真实性（Realism）分数上优于其他方法，最终综合模型在排行榜上达到0.778的Realism分数，接近SOTA。
* **全面性**：实验设置覆盖了五个模块的独立消融实验（如不同分词方法、位置编码策略的对比）和综合性能测试，评估指标包括ADE、minADE、碰撞率、越界率及真实性分数，任务涵盖预测、仿真和规划，较为全面。
* **局限性**：受限于数据集多样性和质量，scaling law在超大数据量或大模型参数下出现过拟合；测试时计算策略（如搜索+聚类）显著增加计算开销（运行时间从0.69s增至11.31s），需权衡效率和效果。

## Further Thoughts

论文提出的系统性模块迁移框架为跨领域技术应用提供了参考，未来可扩展至机器人控制等领域；Global-DRoPE的空间编码方法启发我们在其他时空序列任务中探索多维关系编码的可能性；测试时计算策略（并行生成+搜索+聚类）展示了推理阶段优化的潜力，未来可研究更高效的搜索算法或动态计算预算分配机制，以平衡性能和效率。