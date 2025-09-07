---
title: "DEXOP: A Device for Robotic Transfer of Dexterous Human Manipulation"
pubDatetime: 2025-09-04T17:57:13+00:00
slug: "2025-09-dexop-perioperation"
type: "arxiv"
id: "2509.04441"
score: 0.751800156637276
author: "grok-3-latest"
authors: ["Hao-Shu Fang", "Branden Romero", "Yichen Xie", "Arthur Hu", "Bo-Ruei Huang", "Juan Alvarez", "Matthew Kim", "Gabriel Margolis", "Kavya Anbarasu", "Masayoshi Tomizuka", "Edward Adelson", "Pulkit Agrawal"]
tags: ["Dexterous Manipulation", "Data Collection", "Exoskeleton", "Tactile Sensing", "Robot Learning"]
institution: ["Improbable AI Lab", "Massachusetts Institute of Technology", "UC Berkeley"]
description: "本文提出 DEXOP，一种被动式手部外骨骼系统，通过机械联动和多模态传感器实现人类灵巧操作数据的高效采集，显著提升数据质量和机器人策略性能。"
---

> **Summary:** 本文提出 DEXOP，一种被动式手部外骨骼系统，通过机械联动和多模态传感器实现人类灵巧操作数据的高效采集，显著提升数据质量和机器人策略性能。 

> **Keywords:** Dexterous Manipulation, Data Collection, Exoskeleton, Tactile Sensing, Robot Learning

**Authors:** Hao-Shu Fang, Branden Romero, Yichen Xie, Arthur Hu, Bo-Ruei Huang, Juan Alvarez, Matthew Kim, Gabriel Margolis, Kavya Anbarasu, Masayoshi Tomizuka, Edward Adelson, Pulkit Agrawal

**Institution(s):** Improbable AI Lab, Massachusetts Institute of Technology, UC Berkeley


## Problem Background

机器人灵巧操作（Dexterous Manipulation）是机器人学中的核心挑战之一，机器学习方法依赖大规模数据，但当前数据采集方式（如仿真、人类视频和远程操作）存在显著局限：仿真面临仿真到现实的差距，人类视频难以提取精细交互信息，远程操作缺乏触觉反馈导致效率低下且难以展示精细任务。
论文提出了一种新的数据采集范式‘Perioperation’，旨在通过传感器化人类操作，捕捉丰富的多模态数据（视觉、触觉、动作等），并最大化数据到真实机器人的可迁移性，解决传统方法在自然性和数据质量上的不足。

## Method

*   **核心理念：** 提出 DEXOP，一种被动式手部外骨骼系统，通过机械联动将人类手部动作映射到被动机器人手部，集成高分辨率触觉和视觉传感器，旨在让人类自然地执行灵巧任务，同时确保采集的数据对机器人学习有效。
*   **设计目标与实现：**
    *   **自然性：** 通过高力透明度和关节级本体感觉反馈（Proprioceptive Feedback），让人类操作者能够实时感知机器人手的交互力，克服远程操作中缺乏触觉反馈的局限；机械设计实现人类手姿到机器人手的直接映射，避免远程操作中常见的姿势校正问题。
    *   **数据可迁移性：** 将触觉传感器安装在被动机器人手上，而非人类手上，通过与真实机器人手相同的运动学链和传感器配置，确保采集数据的直接适用性；采用类似 EyeSight Hand 的整手触觉传感系统，捕捉详细的接触力和交互信息。
    *   **任务多样性：** 设计了三种 DEXOP 变体（DEXOP-12、DEXOP-9、DEXOP-7），具有不同自由度，支持多种任务；通过机械增强（如指甲、外展关节、掌垫）扩展能力，覆盖精细指尖操作和整手操作。
*   **技术细节：** 使用四连杆机构实现人类手与机器人手的运动耦合，配备 GelSim(ple) 相机基触觉传感器和关节位置编码器，结合腕部鱼眼相机采集视觉数据；支持与臂部外骨骼（如 AirExo-2）集成，捕捉全局手部位置。
*   **关键创新：** 将人类手与机器人手分离，避免传统触觉手套的低分辨率和穿戴不适问题，同时通过被动式设计降低复杂性和成本。

## Experiment

*   **硬件性能：** DEXOP-7 在力输出（约 60-70N）、工作空间（关节旋转范围与真实机器人手相当）和手指速度（部分关节速度高于真实机器人手）上与 EyeSight Hand 相当，表明其数据采集能力能够匹配机器人执行需求。
*   **数据采集效率：** 用户研究（4 名参与者，240 次试验）表明，DEXOP 在多个任务（如钻孔、灯泡安装、瓶盖开启）中的任务吞吐量显著高于远程操作，例如灯泡安装任务中 DEXOP 平均耗时 11 秒，而远程操作耗时 86 秒，效率提升约 8 倍；钻孔任务中远程操作无一成功，而 DEXOP 平均每分钟完成 6 次。
*   **策略学习效果：** 在双臂灯泡安装任务中，使用 160 个 DEXOP 演示数据结合 40 个远程操作数据的策略，累计成功率（0.513）高于纯远程操作数据（200 个：0.425，100 个：0.355，40 个：0.350）的策略，尤其在精细对齐和触觉感知依赖的步骤（如灯泡插入）表现更优；DEXOP 数据采集速度是远程操作的 2.67 倍，且数据质量更高，避免了远程操作中因缺乏触觉反馈导致的系统性偏差（如过度旋转灯泡）。
*   **实验设置合理性：** 实验涵盖硬件特性、用户体验和策略学习效果，任务设计（如灯泡安装）包含精细操作和双臂协调，较为全面；但存在局限，如外骨骼制造误差需通过混合数据校准，且任务多为结构化环境，缺乏对非结构化环境的测试。

## Further Thoughts

DEXOP 强调数据采集设备与目标机器人硬件的协同设计，这一思路启发我们可以在其他领域探索定制化外骨骼以加速任务特定数据的采集，或将其作为基础模型预训练工具，通过少量真实机器人数据校正误差；此外，触觉与视觉在不同操作阶段的作用差异提示未来可以设计动态模态加权机制，提升策略鲁棒性；最后，DEXOP 的模块化设计和被动式结构为大规模、低成本数据采集提供了可能，是否可以通过开源硬件进一步推动社区协作，加速机器人灵巧操作的发展？