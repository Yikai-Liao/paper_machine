---
title: "GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning"
pubDatetime: 2025-07-01T17:55:04+00:00
slug: "2025-07-glm-multimodal-reasoning"
type: "arxiv"
id: "2507.01006"
score: 0.6303289352756344
author: "grok-3-latest"
authors: ["Guo Wang", "Guobing Gan", "Haomiao Tang", "Jiale Cheng", "Ji Qi", "Junhui Ji", "Lihang Pan", "Shuaiqi Duan", "Weihan Wang", "Yan Wang", "Yean Cheng", "Zehai He", "Zhe Su", "Zhen Yang", "Ziyang Pan", "Aohan Zeng", "Baoxu Wang", "Boyan Shi", "Changyu Pang", "Chenhui Zhang", "Da Yin", "Fan Yang", "Guoqing Chen", "Jiazheng Xu", "Jiali Chen", "Jing Chen", "Jinhao Chen", "Jinghao Lin", "Jinjiang Wang", "Junjie Chen", "Leqi Lei", "Letian Gong", "Leyi Pan", "Mingzhi Zhang", "Qinkai Zheng", "Sheng Yang", "Shi Zhong", "Shiyu Huang", "Shuyuan Zhao", "Siyan Xue", "Shangqin Tu", "Shengbiao Meng", "Tianshu Zhang", "Tianwei Luo", "Tianxiang Hao", "Wenkai Li", "Wei Jia", "Xin Lyu", "Xuancheng Huang", "Yanling Wang", "Yadong Xue", "Yanfeng Wang", "Yifan An", "Yifan Du", "Yiming Shi", "Yiheng Huang", "Yilin Niu", "Yuan Wang", "Yuanchang Yue", "Yuchen Li", "Yutao Zhang", "Yuxuan Zhang", "Zhanxiao Du", "Zhenyu Hou", "Zhao Xue", "Zhengxiao Du", "Zihan Wang", "Wenyi Hong", "Wenmeng Yu", "Xiaotao Gu", "Peng Zhang", "Debing Liu", "Bin Xu", "Juanzi Li", "Minlie Huang", "Yuxiao Dong", "Jie Tang"]
tags: ["LLM", "Vision-Language Model", "Reinforcement Learning", "Curriculum Sampling", "Reasoning", "Pre-Training", "Supervised Fine-Tuning", "Cross-Domain Generalization"]
institution: ["Zhipu AI", "Tsinghua University"]
description: "本文提出 GLM-4.1V-Thinking 模型，通过以推理为中心的训练框架和强化学习课程采样（RLCS），显著提升多模态推理能力，在多个基准上达到最先进的性能。"
---

> **Summary:** 本文提出 GLM-4.1V-Thinking 模型，通过以推理为中心的训练框架和强化学习课程采样（RLCS），显著提升多模态推理能力，在多个基准上达到最先进的性能。 

> **Keywords:** LLM, Vision-Language Model, Reinforcement Learning, Curriculum Sampling, Reasoning, Pre-Training, Supervised Fine-Tuning, Cross-Domain Generalization

**Authors:** Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Wenkai Li, Wei Jia, Xin Lyu, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuxuan Zhang, Zhanxiao Du, Zhenyu Hou, Zhao Xue, Zhengxiao Du, Zihan Wang, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, Jie Tang

**Institution(s):** Zhipu AI, Tsinghua University


## Problem Background

视觉-语言模型（Vision-Language Models, VLMs）在现代智能系统中扮演着重要角色，但当前开源社区缺乏一个在广泛多模态任务上持续优于传统非推理模型的通用多模态推理模型，尤其是在 STEM 问题求解、视频理解、长文档理解等领域，模型的推理能力不足以应对复杂任务。
论文旨在通过构建一个以推理为中心的训练框架，全面提升模型在多模态任务中的理解和推理能力。

## Method

* **整体框架**：论文提出了一种多阶段训练框架，包括大规模预训练、监督微调（Supervised Fine-Tuning, SFT）和强化学习（Reinforcement Learning, RL），以提升模型的多模态推理能力。
* **预训练阶段**：通过构建多样化的多模态数据集，包括图像-文本对、学术语料、OCR 数据、视觉定位数据和视频数据等，为模型奠定强大的基础能力。数据处理涉及多阶段过滤（如启发式过滤、相关性过滤）、概念平衡重采样和事实中心重新描述，确保数据质量和覆盖广度。
* **监督微调阶段**：设计长链式推理（Chain-of-Thought, CoT）数据集，训练模型以标准化的推理格式（如 <think> 和 <answer> 标签）应对多领域任务，为后续强化学习提供‘冷启动’基础，确保模型具备初步的推理风格和人类对齐能力。
* **强化学习阶段**：提出‘强化学习与课程采样’（Reinforcement Learning with Curriculum Sampling, RLCS），通过难度感知的采样策略动态调整训练样本难度，提升训练效率；结合可验证奖励（RLVR）和人类反馈强化学习（RLHF），构建多领域统一奖励系统，针对不同任务（如 STEM、OCR、视频理解）设计特定奖励机制，优化模型表现。
* **模型架构**：采用 ViT 编码器、MLP 投影器和大型语言模型解码器的组合，支持任意分辨率图像和视频输入，通过 2D/3D-RoPE 增强空间理解，通过时间索引 token 提升视频的时间理解能力。
* **训练优化**：包括大批量训练、动态采样扩展（通过比率指数移动平均）、强制回答策略（避免推理过长截断）、移除 KL 和熵损失等，确保训练效率和稳定性。

## Experiment

* **有效性**：GLM-4.1V-9B-Thinking 在 28 个公开基准数据集上进行评估，覆盖通用视觉问答、STEM、OCR & 图表、长文档理解等八大类别；在同规模模型中表现最佳，在 23 个基准上取得领先；与更大规模模型 Qwen2.5-VL-72B 相比，在 18 个基准上表现相当甚至更优；与闭源模型 GPT-4o 相比，在多个挑战性任务（如 MMStar、MathVista）上表现更优。
* **提升显著性**：强化学习带来显著性能提升，部分任务提升高达 7.3%，特别是在 STEM、长文档理解和 GUI 代理任务上表现突出。
* **实验设置**：评估设置全面，涵盖多领域任务，使用一致的工具链和策略（如 vLLM 推理、GPT-4o 评分），确保公平性；唯一不足是部分基准可能接近饱和，难以进一步区分模型能力。
* **开销与效率**：模型在 9B 参数规模下展现出高效率，优于许多更大规模模型，适合资源受限的实际部署场景。

## Further Thoughts

论文中多领域强化学习的跨领域泛化现象令人启发，训练一个领域的数据可以提升其他领域的表现，提示我们可以探索更多模态间的相互强化机制，例如视觉推理是否能提升纯文本推理能力；RLCS 的课程采样策略为动态调整训练难度提供了新思路，未来可应用于其他模型训练场景甚至自适应学习系统；此外，奖励系统设计中对中间推理步骤评估的不足启发我们设计更精细的奖励机制，关注推理过程而不仅是最终结果。