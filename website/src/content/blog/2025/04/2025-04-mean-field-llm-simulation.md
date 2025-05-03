---
title: "MF-LLM: Simulating Collective Decision Dynamics via a Mean-Field Large Language Model Framework"
pubDatetime: 2025-04-30T12:41:51+00:00
slug: "2025-04-mean-field-llm-simulation"
type: "arxiv"
id: "2504.21582"
score: 0.5830897715618675
author: "grok-3-latest"
authors: ["Qirui Mi", "Mengyue Yang", "Xiangning Yu", "Zhiyu Zhao", "Cheng Deng", "Bo An", "Haifeng Zhang", "Xu Chen", "Jun Wang"]
tags: ["LLM", "Social Simulation", "Mean Field Theory", "Feedback Loop", "Fine-Tuning"]
institution: ["Institute of Automation, Chinese Academy of Sciences", "School of Artificial Intelligence, Chinese Academy of Sciences", "Nanyang Technological University", "University of Bristol", "Tianjin University", "Shanghai Jiao Tong University", "Renmin University of China", "University College London"]
description: "本文提出MF-LLM框架，通过均场理论与大型语言模型的结合及基于信息瓶颈的IB-Tune微调方法，显著提升了集体决策动态模拟的保真度和可扩展性。"
---

> **Summary:** 本文提出MF-LLM框架，通过均场理论与大型语言模型的结合及基于信息瓶颈的IB-Tune微调方法，显著提升了集体决策动态模拟的保真度和可扩展性。 

> **Keywords:** LLM, Social Simulation, Mean Field Theory, Feedback Loop, Fine-Tuning

**Authors:** Qirui Mi, Mengyue Yang, Xiangning Yu, Zhiyu Zhao, Cheng Deng, Bo An, Haifeng Zhang, Xu Chen, Jun Wang

**Institution(s):** Institute of Automation, Chinese Academy of Sciences, School of Artificial Intelligence, Chinese Academy of Sciences, Nanyang Technological University, University of Bristol, Tianjin University, Shanghai Jiao Tong University, Renmin University of China, University College London


## Problem Background

预测大型群体随时间调整决策对于理解公众舆论传播、政策冲击反应及紧急情况下的群体动态至关重要。
传统基于代理的模型依赖手工规则，缺乏现实性和泛化能力，而现有基于大型语言模型（LLM）的社会模拟方法虽有潜力，但生成的模拟结果与现实数据存在偏差，尤其是在量化一致性方面，难以支持政策评估或干预规划等任务。

## Method

*   **核心框架：Mean-Field LLM (MF-LLM)**
    *   提出了一种结合均场理论和大型语言模型（LLM）的框架，用于模拟大规模群体的集体决策动态，通过双向反馈循环显式建模个体与群体之间的动态交互。
    *   包含两个主要模块：
        - **策略模型（Policy Model）**：基于LLM，负责根据个体私有状态（如角色描述、偏好）和群体层面的均场信号生成自然语言形式的个体行动，捕捉个体决策的上下文敏感性和表达多样性。
        - **均场模型（Mean Field Model）**：同样基于LLM，通过均场近似从最近的个体状态-行动对中提炼群体信号，并递归更新群体分布，避免直接建模所有成对交互，提高大规模模拟的可扩展性。
    *   框架通过交替运行这两个模块，生成随时间演变的决策轨迹，模拟微观行为与宏观趋势的动态反馈。
*   **训练方法：IB-Tune 算法**
    *   提出了一种基于信息瓶颈（Information Bottleneck）原理的微调方法，旨在提高模拟与现实数据的匹配度。
    *   对均场模型，优化目标是保留与未来个体行动预测相关的群体特征，同时过滤掉无关历史上下文，生成简洁且决策相关的群体信号，通过变分近似和预测似然估计实现信息压缩与预测精度的平衡。
    *   对策略模型，通过负对数似然（Negative Log-Likelihood）监督训练，使其生成的个体行动更贴近真实人类行为，确保个体决策的现实性。
    *   联合优化两个模块，形成自适应的群体信号更新和个体决策生成循环，提升长程模拟的准确性。
*   **关键创新**：
    *   通过均场信号的递归更新机制，缓解了长轨迹输入爆炸问题，支持大规模代理和长时间跨度的模拟。
    *   IB-Tune 确保了群体信号的高效性和预测相关性，避免了过拟合历史模式，提升了模拟的量化保真度。

## Experiment

*   **数据集与评估设置**：基于微博（WEIBO）数据集，包含约5000个真实社交事件，覆盖七个领域（如犯罪、文化、健康等），评估指标包括个体行动的语义维度（如情感、态度、立场）和群体行动分布的相似性（如KL散度、Wasserstein距离、动态时间规整距离等）。
*   **效果显著性**：MF-LLM 在与现实数据分布的匹配度上显著优于基线方法（如State、Recent、Popular、SFT），KL散度降低了47%，在多个语义维度（如情感、立场）和领域上均表现出色，特别是在短、中、长期动态模拟中能够捕捉行为转变的时机和幅度。
*   **鲁棒性与泛化性**：实验展示了MF-LLM 在不同LLM骨干（如GPT-4o-mini、Qwen2-1.5B）上的鲁棒性，并在七个社会领域中实现了一致的性能提升，表明其无需任务特定调整即可泛化。
*   **消融实验**：验证了均场模块和IB-Tune的重要性，去除任一组件会导致性能显著下降（如KL散度增加高达118%），证明了双向反馈和信息瓶颈优化的关键作用。
*   **局限性与权衡**：尽管MF-LLM在模拟保真度上表现优异，但其在单步预测的NLL损失上略逊于SFT基线，表明其在长程一致性与短程精确性之间存在权衡；此外，较小模型在模拟群体多样性时优于较大模型，提示模型规模与输出多样性之间的潜在矛盾。

## Further Thoughts

MF-LLM 将均场理论与LLM结合，为模拟大规模复杂系统（如金融市场、生态系统）提供了可扩展思路；IB-Tune 的信息瓶颈优化方法可扩展至多智能体学习或时间序列预测中，用于提取关键特征；论文发现小模型在模拟群体多样性时表现更优，启发我们探索通过正则化或多样性激励改进大模型输出多样性的方法；此外，引入外生信号提高模拟保真度的思路提示未来可设计动态事件检测和响应机制，使系统自适应处理现实世界的突发变化。