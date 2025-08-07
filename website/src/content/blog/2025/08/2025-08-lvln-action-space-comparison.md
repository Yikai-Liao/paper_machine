---
title: "Following Route Instructions using Large Vision-Language Models: A Comparison between Low-level and Panoramic Action Spaces"
pubDatetime: 2025-08-04T21:45:21+00:00
slug: "2025-08-lvln-action-space-comparison"
type: "arxiv"
id: "2508.02917"
score: 0.46278484665244973
author: "grok-3-latest"
authors: ["Vebjørn Haug Kåsene", "Pierre Lison"]
tags: ["Large Vision-Language Models", "Vision-and-Language Navigation", "Action Space", "Fine-Tuning", "Navigation Performance"]
institution: ["University of Oslo", "Norwegian Computing Center"]
description: "本文通过微调现成大型视觉-语言模型（LVLMs）并对比低级与全景动作空间，验证了 LVLMs 在视觉与语言导航（VLN）任务中的适用性，并揭示全景动作空间在性能上的显著优势。"
---

> **Summary:** 本文通过微调现成大型视觉-语言模型（LVLMs）并对比低级与全景动作空间，验证了 LVLMs 在视觉与语言导航（VLN）任务中的适用性，并揭示全景动作空间在性能上的显著优势。 

> **Keywords:** Large Vision-Language Models, Vision-and-Language Navigation, Action Space, Fine-Tuning, Navigation Performance

**Authors:** Vebjørn Haug Kåsene, Pierre Lison

**Institution(s):** University of Oslo, Norwegian Computing Center


## Problem Background

视觉与语言导航（Vision-and-Language Navigation, VLN）是一个研究领域，旨在让自主机器人根据自然语言指令在未知环境中导航。
本文的出发点是探索现成的大型视觉-语言模型（Large Vision-Language Models, LVLMs）在 VLN 任务中的潜力，尤其是在不进行架构修改或模拟器训练的情况下。
此外，作者关注动作空间（Action Space）选择对导航性能的影响，试图解决两个关键问题：(1) 现成的 LVLMs 是否能有效支持 VLN 任务？(2) 低级动作空间（Low-level Action Space）和全景动作空间（Panoramic Action Space）对性能有何不同影响？

## Method

*   **核心思想:** 通过行为克隆（Behavior Cloning）微调现成的 LVLM（如 Qwen2.5-VL-3B-Instruct），验证其在 VLN 任务中的适用性，并对比两种动作空间的表现。
*   **具体实现:**
    *   **输入设计:** 模型接收多模态输入，包括自然语言指令、当前视觉观察（图像）、历史上下文、当前步数和累计距离等信息，输入以结构化提示（Prompt）形式组织。
    *   **动作空间定义:**
        - **低级动作空间:** 基于自我中心视角（Egocentric View），每次决策基于单一图像，动作包括‘向前移动’（Move）、‘左转’（Left）、‘右转’（Right）和‘停止’（Stop），并引入自动朝向节点调整（Automatically Turn Towards Node）以优化移动方向。
        - **全景动作空间:** 基于 360 度全景图像，模型从一组候选方向（Candidate Directions）中选择一个可导航方向或‘停止’，每个候选方向包含图像、相对角度和距离信息。
    *   **微调策略:** 通过最小化专家动作的负对数似然损失进行训练，仅调整语言模型部分，视觉编码器和跨模态投影层保持冻结，不依赖模拟器训练或强化学习。
    *   **推理方式:** 采用贪婪搜索（Greedy Search），在每一步选择概率最高的动作，无回溯机制。
*   **关键点:** 方法强调通用性，避免架构修改或复杂训练策略，旨在直接评估现成 LVLMs 的能力，同时通过两种动作空间对比揭示视觉输入和决策粒度对导航的影响。

## Experiment

*   **有效性:** 在 R2R 数据集上，全景动作空间模型（Qwen2.5-VL-pano）在测试集上的成功率（Success Rate, SR）达到 41%，显著优于低级动作空间模型（Qwen2.5-VL-low，SR 为 26%），表明全景视角在减少决策步骤和错误累积方面有明显优势。
*   **与基准对比:** 全景模型优于早期基准模型（如 Seq2Seq 的 21% SR 和 Speaker-Follower 的 36% SR），但仍落后于专门为 VLN 设计的最新模型（如 NaviLLM 的 60% SR 和 NavGPT-2 的 72% SR），显示现成 LVLMs 的局限性。
*   **实验设置合理性:** 实验覆盖离线和在线评估，指标包括成功率（SR）、路径长度（PL）、导航误差（NE）等，数据分割（val seen, val unseen, test）全面；但受限于 GPU 内存，批大小仅为 1，且未采用强化学习或学生强制等 VLN 常见训练策略，可能限制性能；此外，全景图像处理方式与传统方法有差异，影响结果可比性。
*   **额外分析:** 低级动作空间下路径步数更长（平均 12-13 步 vs 全景的 6 步），导致错误累积风险增加；低级模型在不同配置（如调整垂直视野或取消自动朝向调整）下的性能变化有限。

## Further Thoughts

论文揭示了动作空间设计对导航性能的显著影响，启发我们思考是否可以通过混合动作空间（结合低级动作的灵活性和全景动作的全局视角）进一步提升性能；此外，现成 LVLMs 在空间推理上的不足提示我们，可以探索引入显式空间结构建模（如图网络或空间注意力机制）以弥补短板；最后，通用模型在资源受限场景下的潜力值得进一步挖掘，或许通过更高效的微调策略（如少样本学习或提示工程）可提升其在 VLN 中的表现。