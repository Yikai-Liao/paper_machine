---
title: "MF-LLM: Simulating Collective Decision Dynamics via a Mean-Field Large Language Model Framework"
pubDatetime: 2025-05-01T15:51:51Z
slug: "2025-04-mean-field-llm-simulation"
type: "arxiv"
id: "2504.21582"
score: 0.8979276198717797
author: "grok-3-latest"
authors: ["Qirui Mi", "Mengyue Yang", "Xiangning Yu", "Zhiyu Zhao", "Cheng Deng", "Bo An", "Haifeng Zhang", "Xu Chen", "Jun Wang"]
tags: ["LLM", "Social Simulation", "Mean Field", "Feedback Loop", "Fine-Tuning"]
institution: ["Institute of Automation, Chinese Academy of Sciences", "School of Artificial Intelligence, Chinese Academy of Sciences", "Nanyang Technological University", "University of Bristol", "Tianjin University", "Shanghai Jiao Tong University", "Renmin University of China", "University College London"]
description: "本文提出MF-LLM框架，通过均值场理论与大语言模型结合，显式建模个体与群体间的双向反馈，显著提升集体决策动态模拟的保真度。"
---

> **Summary:** 本文提出MF-LLM框架，通过均值场理论与大语言模型结合，显式建模个体与群体间的双向反馈，显著提升集体决策动态模拟的保真度。 

> **Keywords:** LLM, Social Simulation, Mean Field, Feedback Loop, Fine-Tuning
> **Recommendation Score:** 0.8979276198717797

**Authors:** Qirui Mi, Mengyue Yang, Xiangning Yu, Zhiyu Zhao, Cheng Deng, Bo An, Haifeng Zhang, Xu Chen, Jun Wang
**Institution(s):** Institute of Automation, Chinese Academy of Sciences, School of Artificial Intelligence, Chinese Academy of Sciences, Nanyang Technological University, University of Bristol, Tianjin University, Shanghai Jiao Tong University, Renmin University of China, University College London

## Problem Background

集体决策动态的模拟在预测公众舆论、政策影响和紧急情况下的群体行为等方面至关重要，但传统基于代理的模型（ABM）依赖手工规则，缺乏现实性和泛化能力，而现有基于大语言模型（LLM）的社会模拟方法虽有潜力，却因未能显式建模个体与群体之间的动态反馈，导致与现实数据的定量匹配不足，难以支持高保真的任务如政策评估或干预规划。

## Method

* **核心思想**：提出MF-LLM（Mean-Field LLM）框架，通过结合均值场理论和LLM的生成能力，显式建模个体决策与群体趋势之间的双向反馈，实现高保真的集体决策动态模拟。
* **模块设计**：
  * **策略模型（Policy Model）**：基于LLM，输入个体的私人状态（如角色描述、偏好）和群体层面的均值场信号，生成个体决策，决策空间为无界的自然语言空间，以捕捉丰富的上下文敏感行为。
  * **均值场模型（Mean Field Model）**：同样基于LLM，从最近的个体状态-行为对中提炼群体分布信号，并递归更新该信号，作为影响后续个体决策的群体反馈，避免直接建模所有成对交互带来的计算复杂度。
  * 两个模块交替运行，形成闭环，模拟个体与群体之间的动态交互过程。
* **优化方法**：提出IB-Tune算法，基于信息瓶颈原理（Information Bottleneck）联合微调两个模型：
  * 对均值场模型，通过优化信息瓶颈目标，保留与未来个体行为预测相关的群体特征，过滤无关历史信息，生成精简且预测性强的群体信号。
  * 对策略模型，通过负对数似然损失（Negative Log-Likelihood）监督训练，使其生成的个体行为更贴近现实数据。
* **关键创新**：通过均值场信号的递归更新，解决大规模交互轨迹的输入爆炸问题，同时利用LLM的生成能力克服传统ABM的手工规则限制。

## Experiment

* **有效性**：在WEIBO数据集（约5000个真实社交事件）上，MF-LLM显著优于基线方法（如State, Recent, Popular, SFT），与现实群体分布的KL散度降低了47%，在多个语义维度（如情感、立场）上表现出更高的保真度。
* **泛化性**：框架在七个不同领域（犯罪、文化、健康等）和四种LLM骨干模型（如GPT-4o-mini, Qwen2-1.5B）上均表现出稳健性能，表明其跨领域和跨模型的适用性。
* **消融研究**：去掉均值场模块或IB-Tune算法会导致性能显著下降（如KL散度增加高达118%），验证了两者对模拟保真度的关键作用。
* **实际应用**：MF-LLM在趋势预测和干预规划中表现出色，例如在谣言传播场景中能够准确预测未来趋势并评估干预效果。
* **实验设置合理性**：实验涵盖短期、中期和长期动态模拟，评估指标（如KL散度、Wasserstein距离、DTW距离、F1分数）从个体行为和群体分布两个层面衡量模拟效果，较为全面；但NLL损失在SFT基线上的表现优于MF-LLM，提示单步预测准确性与长期分布一致性之间存在权衡。

## Further Thoughts

均值场信号作为个体与群体交互的中介，不仅解决了社会模拟中的计算复杂度问题，还可能为其他多智能体系统（如金融市场模拟或交通流量预测）提供新思路；IB-Tune算法基于信息瓶颈原理优化群体信号的预测相关性，或可应用于时间序列预测或个性化推荐等需要提取关键信息的场景；此外，论文发现较小LLM在模拟群体动态时因输出多样性而优于大模型，提示是否可以通过引入噪声或多样性正则化提升大模型在类似任务中的表现。