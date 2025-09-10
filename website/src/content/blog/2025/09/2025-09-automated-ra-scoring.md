---
title: "Automated Radiographic Total Sharp Score (ARTSS) in Rheumatoid Arthritis: A Solution to Reduce Inter-Intra Reader Variation and Enhancing Clinical Practice"
pubDatetime: 2025-09-08T16:21:45+00:00
slug: "2025-09-automated-ra-scoring"
type: "arxiv"
id: "2509.06854"
score: 0.7934810313131055
author: "grok-3-latest"
authors: ["Hajar Moradmand", "Lei Ren"]
tags: ["Deep Learning", "Medical Imaging", "Automated Scoring", "Joint Detection", "Rheumatoid Arthritis"]
institution: ["University of Maryland School of Medicine"]
description: "本文提出 ARTSS 框架，利用深度学习自动化评估类风湿关节炎手部 X 光图像的 TSS 分数，显著降低评分变异性并提高临床效率，同时创新解决关节数量变异问题。"
---

> **Summary:** 本文提出 ARTSS 框架，利用深度学习自动化评估类风湿关节炎手部 X 光图像的 TSS 分数，显著降低评分变异性并提高临床效率，同时创新解决关节数量变异问题。 

> **Keywords:** Deep Learning, Medical Imaging, Automated Scoring, Joint Detection, Rheumatoid Arthritis

**Authors:** Hajar Moradmand, Lei Ren

**Institution(s):** University of Maryland School of Medicine


## Problem Background

类风湿关节炎（RA）是一种慢性自身免疫疾病，临床上常用 Total Sharp/van der Heijde Score (TSS) 评估关节损伤，但手动评分耗时长、主观性强，存在显著的读者间和读者内变异性，限制了其临床应用效果；此外，RA 患者手部 X 光图像中关节数量可能因严重侵蚀而变化，传统计算机模型难以处理此类变异性，导致研究中常排除复杂病例，影响模型泛化性。

## Method

* **整体框架**：提出 Automated Radiographic Total Sharp Score (ARTSS) 框架，通过深度学习技术自动化分析手部 X 光图像，分为四个阶段：图像预处理、手部分割、关节识别和 TSS 预测。
* **图像预处理与方向校正**：使用 ResNet50 模型将图像统一调整至 90 度方向，并进行尺寸调整和归一化，确保输入一致性，增强模型对不同设备和患者姿势的适应性。
* **手部分割**：采用 U-Net 架构进行语义分割，通过高斯滤波、小波变换去噪、阈值处理和形态学操作生成精确手部掩码，从背景中分离出手部区域。
* **关节识别**：利用 YOLOv7 目标检测算法识别关键关节（如近端指间关节、掌指关节、腕关节），通过左右手图像分离和数据增强解决识别混淆问题，提升检测精度。
* **TSS 预测**：测试多种深度学习模型（包括 VGG16、VGG19、ResNet50、DenseNet201、EfficientNetB0 和 Vision Transformer）预测 TSS 分数；针对关节数量变异，设计填充和掩码技术确保输入序列长度一致，模型仅关注有效数据。
* **辅助技术**：采用数据增强（旋转、平移、翻转、亮度调整）提高模型泛化能力，使用 Huber 损失函数优化回归任务，平衡小误差和大误差的影响。

## Experiment

* **数据集与设置**：使用 970 名患者的公开手部 X 光图像，采用 3 折交叉验证（每折 452 训练、227 验证样本），并在 291 名未见过受试者上进行外部测试，确保结果泛化性；评价指标包括 IoU、MAP、MAE、RMSE 和 Huber 损失。
* **手部分割结果**：U-Net 模型在测试集上的 IoU 达到 0.94，显示出极高的分割精度。
* **关节识别结果**：YOLOv7 模型准确率达 99%，显著优于文献中其他研究的 87%-94.5%，表明其在复杂手部图像中的强大适应性。
* **TSS 预测结果**：Vision Transformer (ViT) 表现最佳，Huber 损失仅为 0.87，MAE 和 RMSE 分别为 0.95 和 0.93，远优于其他模型（如 EfficientNetB0 的 Huber 损失为 8.17）；VGG16 和 VGG19 次优，ResNet50 和 DenseNet201 误差较高。
* **综合评价**：实验设置全面，涵盖训练、验证和外部测试，指标合理，ARTSS 框架在减少评分变异性和提高效率方面效果显著，但数据集来自单一中心，图像角度和部位限制可能影响泛化性。

## Further Thoughts

论文提出的填充和掩码技术为处理医疗影像中变异性数据提供了新思路，可推广至其他不规则影像任务；多模型比较揭示 Transformer 架构在医疗影像回归中的潜力，值得探索其在其他疾病评估中的应用；ARTSS 的自动化理念可扩展至资源有限地区的临床辅助决策，未来可结合多中心、多模态数据（如足部影像或 MRI）进一步提升模型鲁棒性和适用性。