---
title: "Doctoral Thesis: Geometric Deep Learning For Camera Pose Prediction, Registration, Depth Estimation, and 3D Reconstruction"
pubDatetime: 2025-09-02T01:35:44+00:00
slug: "2025-09-geometric-deep-learning-3d-vision"
type: "arxiv"
id: "2509.01873"
score: 0.5016625424091598
author: "grok-3-latest"
authors: ["Xueyang Kang"]
tags: ["Geometric Deep Learning", "Camera Pose Estimation", "Point Cloud Registration", "Depth Estimation", "3D Reconstruction"]
institution: ["KU Leuven", "The University of Melbourne"]
description: "本文通过将几何约束与深度学习结合，系统性提升了相机姿态估计、点云配准、深度估计和 3D 重建任务的精度与鲁棒性，为文化遗产数字化和虚拟现实等应用提供了高效解决方案。"
---

> **Summary:** 本文通过将几何约束与深度学习结合，系统性提升了相机姿态估计、点云配准、深度估计和 3D 重建任务的精度与鲁棒性，为文化遗产数字化和虚拟现实等应用提供了高效解决方案。 

> **Keywords:** Geometric Deep Learning, Camera Pose Estimation, Point Cloud Registration, Depth Estimation, 3D Reconstruction

**Authors:** Xueyang Kang

**Institution(s):** KU Leuven, The University of Melbourne


## Problem Background

3D 视觉技术在现实应用中至关重要，但传统方法如结构从运动（SfM）和同时定位与建图（SLAM）在非结构化环境中因特征模糊和噪声而失效；同时，深度学习直接处理 3D 数据面临高维性和标注数据稀缺的挑战。
本文旨在通过结合几何约束与深度学习，解决相机姿态估计、点云配准、深度估计和 3D 重建中的关键问题，提升模型在复杂环境中的精度和鲁棒性。

## Method

*   **相机姿态估计（Chapter 3）:** 提出一种基于视觉的姿态跟踪系统，利用自然环境中的天际线和地面平面作为几何线索，通过轻量级 ResNet-18 网络在嵌入式设备上实时分割图像为地面和天空区域；采用几何框架从分割结果中推断相机旋转角度；设计自适应粒子滤波器在多分辨率流形上融合视觉线索和惯性测量单元（IMU）数据，确保在挑战性户外环境中的鲁棒性。
*   **点云配准（Chapter 4）:** 引入基于 2D 表面元素（Surfel）的 SE(3)-等变深度学习框架，从 RGB-D 或 LiDAR 数据提取表面元素表示；通过适配的 E2PN 编码器学习点位置和方向特征；使用结构化 Huber 损失优化姿态回归，提升对噪声和低重叠数据的鲁棒性。
*   **深度估计（Chapter 5）:** 提出 Transformer 网络 FocDepthFormer，结合自注意力机制和 LSTM 循环模块处理任意长度的焦距堆栈图像；通过多尺度卷积编码器提取焦点/离焦线索，嵌入焦距几何信息到训练损失中，提升深度预测精度和灵活性。
*   **3D 重建（Chapter 6）:** 提出基于隐式符号距离场（SDF）的重建框架，通过小波变换深度特征条件化隐式模型，结合三平面（Triplane）特征投影和融合，增强多尺度几何细节表示；使用 2D UNet 融合模块优化 SDF 预测，生成高保真 3D 表面。

## Experiment

*   **相机姿态估计:** 在真实无人机和屋顶环境中测试，结果显示滚转和俯仰角度的均方根误差（RMSE）显著低于基线方法，在自然环境下的稳定性提升明显；实验设置涵盖动态场景，较为全面。
*   **点云配准:** 在 3DMatch 和 KITTI 数据集上评估，配准精度和鲁棒性超越现有技术，尤其在低内点比例场景中表现突出；消融研究验证了各组件贡献，实验设计合理。
*   **深度估计:** FocDepthFormer 在 DDFF 12-Scene、FOD500 和 LightField4D 数据集上取得最先进性能，深度预测精度和输入长度适应性显著提升；实验覆盖多种场景，设置科学。
*   **3D 重建:** 在 DTU、Tanks and Temples 及文化遗产数据集上，Chamfer 距离（CD）指标优于基线，重建表面细节更丰富；消融研究表明小波编码器和三平面融合对细节保留至关重要，实验全面且具说服力。

## Further Thoughts

几何约束与深度学习的结合提供了提升模型解释性和鲁棒性的新思路，可推广至机器人导航或增强现实领域；FocDepthFormer 处理任意长度输入的设计启发灵活架构在序列数据任务中的应用；3D 重建中多尺度特征表示（如小波变换和三平面融合）对细节保留的重要性，可应用于图像超分辨率或医学影像重建等领域。