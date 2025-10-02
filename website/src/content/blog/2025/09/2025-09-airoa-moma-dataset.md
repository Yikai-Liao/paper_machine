---
title: "AIRoA MoMa Dataset: A Large-Scale Hierarchical Dataset for Mobile Manipulation"
pubDatetime: 2025-09-29T16:51:47+00:00
slug: "2025-09-airoa-moma-dataset"
type: "arxiv"
id: "2509.25032"
score: 0.7189119025102961
author: "grok-3-latest"
authors: ["Ryosuke Takanami", "Petr Khrapchenkov", "Shu Morikuni", "Jumpei Arima", "Yuta Takaba", "Shunsuke Maeda", "Takuya Okubo", "Genki Sano", "Satoshi Sekioka", "Aoi Kadoya", "Motonari Kambara", "Naoya Nishiura", "Haruto Suzuki", "Takanori Yoshimoto", "Koya Sakamoto", "Shinnosuke Ono", "Hu Yang", "Daichi Yashima", "Aoi Horo", "Tomohiro Motoda", "Kensuke Chiyoma", "Hiroshi Ito", "Koki Fukuda", "Akihito Goto", "Kazumi Morinaga", "Yuya Ikeda", "Riko Kawada", "Masaki Yoshikawa", "Norio Kosuge", "Yuki Noguchi", "Kei Ota", "Tatsuya Matsushima", "Yusuke Iwasawa", "Yutaka Matsuo", "Tetsuya Ogata"]
tags: ["Mobile Manipulation", "Multimodal Data", "Hierarchical Learning", "Contact-Rich Tasks", "Long-Horizon Tasks"]
institution: ["The University of Tokyo", "AI Robot Association (AIRoA)", "Toyota Motor Corporation", "Telexistence, Inc.", "National Institute of Advanced Industrial Science and Technology (AIST)", "Waseda University"]
description: "本文介绍了 AIRoA MoMa 数据集，一个大规模、多模态、层次化的移动操作数据集，填补了接触丰富和长程任务领域的空白，为下一代通用机器人研究提供了关键基准。"
---

> **Summary:** 本文介绍了 AIRoA MoMa 数据集，一个大规模、多模态、层次化的移动操作数据集，填补了接触丰富和长程任务领域的空白，为下一代通用机器人研究提供了关键基准。 

> **Keywords:** Mobile Manipulation, Multimodal Data, Hierarchical Learning, Contact-Rich Tasks, Long-Horizon Tasks

**Authors:** Ryosuke Takanami, Petr Khrapchenkov, Shu Morikuni, Jumpei Arima, Yuta Takaba, Shunsuke Maeda, Takuya Okubo, Genki Sano, Satoshi Sekioka, Aoi Kadoya, Motonari Kambara, Naoya Nishiura, Haruto Suzuki, Takanori Yoshimoto, Koya Sakamoto, Shinnosuke Ono, Hu Yang, Daichi Yashima, Aoi Horo, Tomohiro Motoda, Kensuke Chiyoma, Hiroshi Ito, Koki Fukuda, Akihito Goto, Kazumi Morinaga, Yuya Ikeda, Riko Kawada, Masaki Yoshikawa, Norio Kosuge, Yuki Noguchi, Kei Ota, Tatsuya Matsushima, Yusuke Iwasawa, Yutaka Matsuo, Tetsuya Ogata

**Institution(s):** The University of Tokyo, AI Robot Association (AIRoA), Toyota Motor Corporation, Telexistence, Inc., National Institute of Advanced Industrial Science and Technology (AIST), Waseda University


## Problem Background

当前机器人研究面临的一个核心挑战是开发能够在非结构化人类环境中执行复杂任务的通用机器人，而 Vision-Language-Action (VLA) 模型的性能受限于现有数据集的规模、多样性和质量，尤其是在移动操作（Mobile Manipulation）、接触丰富任务（Contact-Rich Tasks）和长程任务（Long-Horizon Tasks）方面的不足，缺乏同步力-扭矩数据和层次化标注，导致模型难以泛化到真实家庭场景。

## Method

* **数据采集平台与环境**：使用 Toyota Human Support Robot (HSR) 在模拟家庭环境（厨房、客厅、浴室等）中采集数据，通过随机化物体位置、光照条件和机器人初始位置增加数据多样性。
* **多模态数据采集**：提供同步的多模态数据流（30Hz 采样率），包括视觉数据（头戴和腕部双视角 RGB 图像，480x640x3）、本体感觉数据（关节角度、速度、末端执行器姿态）、力-扭矩数据（六轴腕部信号：Fx, Fy, Fz, Mx, My, Mz）以及遥操作控制信号，用于支持接触丰富任务和动作表征研究。
* **遥操作系统设计**：开发 THSR（Teleoperation system for HSR），一种一对一关节映射的领导-跟随系统，通过直接传递关节命令避免逆运动学计算，结合 Joy-Con 控制器辅助移动基座和头部操作，提升操作直观性和安全性，18 名操作员经过培训后完成数据采集。
* **层次化标注框架**：提出两层标注结构，包括高层次的短程任务（Short Horizon Task, SHT，如‘烤面包’）和低层次的原始动作（Primitive Action, PA，如‘打开烤箱’），支持任务分解、层次化学习和细粒度错误分析。
* **数据处理与标准化**：通过多阶段同步（不同传感器频率统一到 30Hz）、过滤（移除无效帧和异常值）、隐私保护（使用 YOLO 检测移除含人类图像片段）确保数据质量，并标准化为 LeRobot v2.1 格式，与现有 VLA 模型兼容，便于社区使用。

## Experiment

* **数据集规模与统计**：AIRoA MoMa 数据集包含 25,469 个片段，总时长约 94 小时，平均片段时长 13 秒，覆盖 7 个主要家庭任务（如‘挂毛巾’‘煮咖啡’）和 40 多个子任务，总数据量约 92 GB。
* **数据分布与质量**：统计显示基础操作（如‘抓取’‘放置’）占主导，任务时长多集中在 4-12 秒，适合训练基础反应性策略；失败案例占比约 6.6%，为错误检测和恢复研究提供素材。
* **实验设置合理性**：数据采集环境模拟真实家庭场景，多模态数据同步采样，层次化标注和失败案例记录设计全面；处理流程（同步、过滤、标准化）严谨，确保数据质量和可用性。
* **局限性**：由于论文聚焦数据集构建，未提供具体模型训练结果，无法直接评估对模型性能的提升效果，但其规模和设计为后续研究奠定了坚实基础。

## Further Thoughts

层次化标注框架（SHT 和 PA）为任务分解和错误分析提供了新思路，未来是否可以扩展到多层次标注，结合环境上下文增强规划能力？失败案例的系统性记录启发我们设计主动学习机制，让机器人从错误中优化策略；多模态数据（尤其是力-扭矩信号）的整合为接触丰富任务提供了新视角，是否可以进一步引入音频或触觉模态提升感知能力？此外，THSR 遥操作系统的直观控制设计是否可以推广到其他平台，或结合虚拟现实技术进一步优化操作体验？