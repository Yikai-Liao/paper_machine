---
title: "Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots"
pubDatetime: 2025-09-02T17:29:38+00:00
slug: "2025-09-camera-depth-models"
type: "arxiv"
id: "2509.02530"
score: 0.5680469111031957
author: "grok-3-latest"
authors: ["Minghuan Liu", "Zhengbang Zhu", "Xiaoshen Han", "Peng Hu", "Haotong Lin", "Xinyao Li", "Jingxiao Chen", "Jiafeng Xu", "Yichu Yang", "Yunfeng Lin", "Xinghang Li", "Yong Yu", "Weinan Zhang", "Tao Kong", "Bingyi Kang"]
tags: ["Robotic Manipulation", "Depth Perception", "Sim-to-Real", "Geometry Information"]
institution: ["ByteDance Seed", "Shanghai Jiao Tong University", "Zhejiang University", "Tsinghua University"]
description: "本文提出相机深度模型（CDMs），通过提升深度相机的几何感知精度，成功弥合仿真与真实世界之间的几何差距，使机器人操作策略在真实环境中实现高成功率的零样本转移。"
---

> **Summary:** 本文提出相机深度模型（CDMs），通过提升深度相机的几何感知精度，成功弥合仿真与真实世界之间的几何差距，使机器人操作策略在真实环境中实现高成功率的零样本转移。 

> **Keywords:** Robotic Manipulation, Depth Perception, Sim-to-Real, Geometry Information

**Authors:** Minghuan Liu, Zhengbang Zhu, Xiaoshen Han, Peng Hu, Haotong Lin, Xinyao Li, Jingxiao Chen, Jiafeng Xu, Yichu Yang, Yunfeng Lin, Xinghang Li, Yong Yu, Weinan Zhang, Tao Kong, Bingyi Kang

**Institution(s):** ByteDance Seed, Shanghai Jiao Tong University, Zhejiang University, Tsinghua University


## Problem Background

机器人操作（Robotic Manipulation）主要依赖2D RGB图像进行技能学习，但这种方法在泛化能力上存在显著不足。
相比之下，人类在3D世界中更依赖几何信息（如距离、形状）而非纹理来与物体交互，而深度相机虽能提供3D信息，却因输出质量差、噪声多、模式失效等问题在真实世界中表现不佳。
论文试图解决的关键问题是提升深度相机的几何感知精度，弥合模拟环境与真实世界之间的几何差距（Sim-to-Real Geometry Gap），从而实现更鲁棒的机器人操作。

## Method

*   **核心思想:** 提出相机深度模型（Camera Depth Models, CDMs），作为深度相机的插件，通过结合RGB图像和原始深度信号，输出去噪且精确的度量深度（Metric Depth），以提升几何感知能力。
*   **模型架构:** 采用双分支Vision Transformer（ViT）架构，分别提取RGB图像的语义信息和深度图像的尺度信息，通过多头注意力机制（Multi-Head Attention, MHA）融合特征，最终使用DPT解码器生成高质量深度图。
*   **数据合成与噪声建模:** 开发神经数据引擎，通过模拟深度相机的噪声模式（包括值噪声和孔洞噪声）生成高质量配对数据；提出引导滤波（Guided Filter）方法解决合成噪声的尺度不匹配问题。
*   **数据集构建:** 构建ByteCameraDepth数据集，包含7种深度相机、10种模式的17万+ RGB-深度对，用于训练噪声模型和CDMs。
*   **训练策略:** 使用L1损失和梯度损失优化模型，确保边缘深度精度，同时初始化ViT编码器权重以提升收敛性。
*   **应用方式:** CDMs作为相机与策略之间的插件，在推理时实时生成清洁深度图像，支持机器人操作任务。

## Experiment

*   **深度预测性能:** 在Hammer数据集的零样本评估中，CDMs显著优于基线方法（如PromptDA和PriorDA），L1误差和RMSE等指标大幅降低，尤其在无需孔洞填充预处理的情况下仍保持高精度，表明其深度预测接近模拟环境水平。
*   **模仿学习任务:** 在真实世界的两个拾取放置任务（Toothpaste-and-Cup和Stack-Bowls）中，使用CDMs生成的深度数据显著提升策略成功率（例如Stack-Bowls任务成功率从0/15提升至9/15），并展现了对不同尺寸物体的泛化能力。
*   **零样本仿真到真实（Sim-to-Real）操作:** 在两个长程任务（Kitchen和Canteen）中，使用CDMs的策略在真实机器人上实现了与模拟环境接近的成功率（例如Kitchen任务成功率接近90%），无需真实世界微调或噪声添加，验证了其在弥合sim-to-real差距上的有效性。
*   **实验设置合理性:** 实验涵盖静态深度预测、真实世界模仿学习和sim-to-real转移任务，数据和任务设计全面，能够充分验证方法效果；唯一不足是推理延迟较高（约0.151秒），可能影响实时性，但可通过优化改进。

## Further Thoughts

CDMs针对特定深度相机定制模型的思路启发我们，未来的AI模型可能需要更多考虑硬件特性，而非一味追求通用性，尤其在机器人感知领域；
通过噪声建模和数据合成利用模拟数据解决真实数据不足的策略，可推广至自动驾驶或医疗影像等领域；
强调3D几何信息在机器人操作中的核心作用，提示未来策略可能需更多依赖3D感知，而非单纯2D视觉输入；
发散思考：是否可结合多模态数据（如RGB-D与触觉）进一步提升感知精度，或利用大型基础模型预训练后针对硬件微调？