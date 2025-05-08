---
title: "Enhancing LLMs' Clinical Reasoning with Real-World Data from a Nationwide Sepsis Registry"
pubDatetime: 2025-05-05T15:23:47+00:00
slug: "2025-05-clinical-reasoning-sepsis"
type: "arxiv"
id: "2505.02722"
score: 0.49632819524431665
author: "grok-3-latest"
authors: ["Junu Kim", "Chaeeun Shim", "Sungjin Park", "Su Yeon Lee", "Gee Young Suh", "Chae-Man Lim", "Seong Jin Choi", "Song Mi Moon", "Kyoung-Ho Song", "Eu Suk Kim", "Hong Bin Kim", "Sejoong Kim", "Chami Im", "Dong-Wan Kang", "Yong Soo Kim", "Hee-Joon Bae", "Sung Yoon Lim", "Han-Gil Jeong", "Edward Choi"]
tags: ["LLM", "Clinical Reasoning", "Real-World Data", "Reinforcement Learning", "Generalization"]
institution: ["Korea Advanced Institute of Science and Technology", "Asan Medical Center, University of Ulsan College of Medicine", "Samsung Medical Center, Sungkyunkwan University School of Medicine", "Seoul National University Bundang Hospital, Seoul National University College of Medicine"]
description: "本文通过真实世界脓毒症登记数据训练大型语言模型，显著提升其临床推理能力，并展示跨任务、跨疾病的泛化性，为通用临床推理模型的发展奠定基础。"
---

> **Summary:** 本文通过真实世界脓毒症登记数据训练大型语言模型，显著提升其临床推理能力，并展示跨任务、跨疾病的泛化性，为通用临床推理模型的发展奠定基础。 

> **Keywords:** LLM, Clinical Reasoning, Real-World Data, Reinforcement Learning, Generalization

**Authors:** Junu Kim, Chaeeun Shim, Sungjin Park, Su Yeon Lee, Gee Young Suh, Chae-Man Lim, Seong Jin Choi, Song Mi Moon, Kyoung-Ho Song, Eu Suk Kim, Hong Bin Kim, Sejoong Kim, Chami Im, Dong-Wan Kang, Yong Soo Kim, Hee-Joon Bae, Sung Yoon Lim, Han-Gil Jeong, Edward Choi

**Institution(s):** Korea Advanced Institute of Science and Technology, Asan Medical Center, University of Ulsan College of Medicine, Samsung Medical Center, Sungkyunkwan University School of Medicine, Seoul National University Bundang Hospital, Seoul National University College of Medicine


## Problem Background

大型语言模型（LLMs）在真实世界临床实践中的推理能力有限，主要由于训练过程中缺乏对真实临床数据的充分暴露。
这种不足导致模型在处理罕见疾病、遵循临床指南以及解释结构化患者数据时表现不佳，限制了其在实际医疗场景中的应用。

## Method

*   **核心思想:** 通过利用真实世界临床数据（如全国性脓毒症登记数据），增强大型语言模型的临床推理能力，开发出名为*C-Reason*的模型。
*   **数据处理与问题构建:** 从脓毒症登记数据中提取患者特征-值对，通过掩码单个特征值生成多选题，促使模型基于剩余信息推断掩码值，这种去噪任务旨在培养模型对特征间关系的理解。
*   **训练策略:** 采用Group Relative Policy Optimization (GRPO)算法进行强化学习，模型生成多个推理过程，并根据推理是否导致正确答案分配奖励，优化推理质量。
*   **提示技术:** 使用零-shot Chain-of-Thought (CoT)提示，鼓励模型在回答前逐步推理，提升推理过程的可解释性。
*   **模型微调:** 对基础模型Phi-4进行全参数微调，确保模型充分适应临床数据特性，同时避免依赖外部强大模型，增强方法的可扩展性。

## Experiment

*   **有效性:** 在脓毒症登记数据的测试集上，*C-Reason*显著优于基础模型Phi-4，例如去噪任务平均准确率从0.712提升至0.864，增幅明显。
*   **泛化能力:** 模型在不同队列（如MIMIC-III数据集）、不同任务（如开放性抗生素使用咨询）以及不同疾病（如急性肾损伤和中风）上均展现了改进的推理能力，表明其临床推理能力具有跨领域、跨任务的泛化性。
*   **专家评估:** 专家评估显示，*C-Reason*的推理逻辑更受临床专家青睐，尤其在初始经验性治疗适当性任务中胜率显著（p<0.0001）。
*   **实验设置合理性:** 实验覆盖多个数据集、任务类型（去噪、预测、开放性生成）和疾病领域，评估指标包括准确率、F1分数及专家偏好，设计较为全面；但部分任务（如1年MACE预测）F1分数下降，可能与数据分布差异有关。

## Further Thoughts

真实世界临床数据的训练价值启发我们探索更多非公开多疾病数据集的使用，以构建通用临床推理模型；此外，方法的可扩展性设计提示类似规则化问题生成策略可能适用于其他领域，如金融或法律推理；最后，模型作为动态交互‘推理伙伴’的潜力表明未来AI系统可在实时临床决策中提供假设探索和指南支持，增强医生的决策能力。