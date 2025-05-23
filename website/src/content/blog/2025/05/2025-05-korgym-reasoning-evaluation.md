---
title: "KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation"
pubDatetime: 2025-05-20T16:06:32+00:00
slug: "2025-05-korgym-reasoning-evaluation"
type: "arxiv"
id: "2505.14552"
score: 0.5336392106613023
author: "grok-3-latest"
authors: ["Jiajun Shi", "Jian Yang", "Jiaheng Liu", "Ge Zhang", "Wenhao Huang", "Xingyuan Bu", "Jiangjie Chen", "Junting Zhou", "Kaijing Ma", "Zhoufutu Wen", "Bingli Wang", "Yancheng He", "Liang Song", "Hualei Zhu", "Shilong Li", "Xingjian Wang", "Wei Zhang", "Ruibin Yuan", "Yifan Yao", "Wenjun Yang", "Yunli Wang", "Siyuan Fang", "Siyu Yuan", "Qianyu He", "Xiangru Tang", "Yingshui Tan", "Wangchunshu Zhou", "Zhaoxiang Zhang", "Zhoujun Li"]
tags: ["LLM", "Reasoning", "Evaluation Platform", "Reinforcement Learning", "Multimodal Tasks"]
institution: ["ByteDance Seed", "M-A-P", "Beihang University"]
description: "本文提出 KORGym 平台，通过超过 50 个多模态游戏和强化学习支持，全面评估大型语言模型的推理能力，揭示模型行为模式并填补现有基准测试的空白。"
---

> **Summary:** 本文提出 KORGym 平台，通过超过 50 个多模态游戏和强化学习支持，全面评估大型语言模型的推理能力，揭示模型行为模式并填补现有基准测试的空白。 

> **Keywords:** LLM, Reasoning, Evaluation Platform, Reinforcement Learning, Multimodal Tasks

**Authors:** Jiajun Shi, Jian Yang, Jiaheng Liu, Ge Zhang, Wenhao Huang, Xingyuan Bu, Jiangjie Chen, Junting Zhou, Kaijing Ma, Zhoufutu Wen, Bingli Wang, Yancheng He, Liang Song, Hualei Zhu, Shilong Li, Xingjian Wang, Wei Zhang, Ruibin Yuan, Yifan Yao, Wenjun Yang, Yunli Wang, Siyuan Fang, Siyu Yuan, Qianyu He, Xiangru Tang, Yingshui Tan, Wangchunshu Zhou, Zhaoxiang Zhang, Zhoujun Li

**Institution(s):** ByteDance Seed, M-A-P, Beihang University


## Problem Background

当前大型语言模型（LLMs）的推理能力评估方法多为领域特定，受到预训练数据的强烈影响，难以全面反映模型的通用推理能力。
现有游戏基准测试存在单轮交互、对手动态干扰等问题，无法有效评估长期规划和纯推理能力。
因此，作者提出 KORGym 平台，旨在通过多样化的游戏任务和多轮互动，提供一个动态、知识正交的评估框架，解决现有基准测试的局限性。

## Method

*   **平台设计:** KORGym 是一个动态游戏平台，包含四个核心模块：
    *   **推理模块（Inference Module）**：管理模型推理过程，支持异步加速和中间结果保存。
    *   **游戏交互模块（Game Interaction Module）**：封装游戏环境和交互接口，包括初始化环境（generate）、渲染游戏板（print board）和更新状态（verify）等功能。
    *   **评估模块（Evaluation Module）**：计算和输出最终评分，采用三种评分规则（二元评分、比例评分、累积评分）以适应不同游戏类型。
    *   **通信模块（Communication Module）**：负责参数解析、模块间通信和数据传输，确保系统运行流畅。
*   **任务设计:** 平台包含超过 50 个文本和视觉游戏，覆盖六个推理维度（数学与逻辑推理、控制交互推理、谜题推理、空间与几何推理、策略推理和多模态推理），包括传统谜题（如 Sudoku）、经典视频游戏（如 Tetris）、博弈论挑战（如 Trust Evolution）和多模态任务（如 Jigsaw Puzzle）。
*   **评估方法:** 引入能力维度聚合平均值（Capability Dimension Aggregated Mean），通过对原始评分进行对数变换和归一化处理，消除游戏难度差异和异常值的影响，确保评估的公平性。
*   **支持功能:** 平台支持多轮交互、强化学习（RL）集成和难度可配置，提供标准化 API 和奖励信号，适用于长期规划和策略学习研究。

## Experiment

*   **有效性:** 实验评估了 19 个 LLM 和 8 个视觉语言模型（VLM），结果显示闭源模型（如 O3-mini 平均得分 82%，Gemini-2.5-pro 79%）在整体推理性能上显著优于开源模型（如 Qwen2.5-7B-Instruct 仅 8%）。
*   **模型系列一致性:** 同一模型系列表现出相似的强弱模式，例如 GPT 系列在空间推理上表现突出（O3-mini 94%），Gemini 系列在数学和谜题推理上领先（Gemini-2.5-pro 数学推理 63%，谜题推理 93%）。
*   **多模态影响:** 文本版本游戏的平均得分普遍高于视觉版本，尤其在数学相关任务中，表明当前 VLM 的视觉推理能力仍有限，但部分闭源 VLM（如 Gemini-2.5-Pro）在视觉任务上表现优于文本任务，显示出较强的多模态整合能力。
*   **强化学习提升:** 强化学习显著提升了推理能力，例如 Doubao-1.5-thinking-pro 在 RL 训练后平均得分达 72%，在谜题推理上达到 84%，超越 O1 和 O3-mini，且在不同维度上表现更均衡。
*   **实验设置合理性:** 实验采用零样本提示，覆盖单轮和多轮游戏，设置 50 个独立游戏实例以确保结果稳健；同时评估了模型规模、架构和推理范式的影响，设置较为全面，但多模态任务数量较少（仅 9 个），可能限制了对视觉推理能力的深入分析。

## Further Thoughts

‘知识正交性’的概念为设计公平的推理评估基准提供了新思路，未来可以探索更多与预训练知识解耦的任务设计，测试模型的纯推理能力。
强化学习在多轮交互中的潜力值得进一步研究，尤其是在复杂策略规划任务中，RL 可能通过动态调整模型行为显著提升性能。
模型在推理范式上的偏好差异（如 Gemini-2.5-Pro 倾向代码范式，O3-mini 偏好数学和自然语言范式）启发我们可以通过定制化提示或训练策略，针对特定任务优化模型表现。