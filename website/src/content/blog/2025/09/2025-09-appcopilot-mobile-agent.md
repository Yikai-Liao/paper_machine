---
title: "AppCopilot: Toward General, Accurate, Long-Horizon, and Efficient Mobile Agent"
pubDatetime: 2025-09-02T15:48:21+00:00
slug: "2025-09-appcopilot-mobile-agent"
type: "arxiv"
id: "2509.02444"
score: 0.7072520340164127
author: "grok-3-latest"
authors: ["Jingru Fan", "Yufan Dang", "Jingyao Wu", "Huatao Li", "Runde Yang", "Xiyuan Yang", "Yuheng Wang", "Zhong Zhang", "Yaxi Lu", "Yankai Lin", "Zhiyuan Liu", "Dahai Li", "Chen Qian"]
tags: ["LLM", "Multimodal Model", "Mobile Agent", "Task Planning", "Multi-Agent Collaboration"]
institution: ["Shanghai Jiao Tong University", "Tsinghua University", "Renmin University of China", "Modelbest Inc."]
description: "本文提出AppCopilot系统，通过多模态多代理设计和全栈闭环架构，显著提升移动代理在泛化能力、操作精度、长程任务规划和设备端效率方面的表现，为通用数字助手的发展提供了参考架构。"
---

> **Summary:** 本文提出AppCopilot系统，通过多模态多代理设计和全栈闭环架构，显著提升移动代理在泛化能力、操作精度、长程任务规划和设备端效率方面的表现，为通用数字助手的发展提供了参考架构。 

> **Keywords:** LLM, Multimodal Model, Mobile Agent, Task Planning, Multi-Agent Collaboration

**Authors:** Jingru Fan, Yufan Dang, Jingyao Wu, Huatao Li, Runde Yang, Xiyuan Yang, Yuheng Wang, Zhong Zhang, Yaxi Lu, Yankai Lin, Zhiyuan Liu, Dahai Li, Chen Qian

**Institution(s):** Shanghai Jiao Tong University, Tsinghua University, Renmin University of China, Modelbest Inc.


## Problem Background

移动代理（Mobile Agent）作为大型语言模型和多模态基础模型的延伸，面临泛化能力不足、单步操作精度不高、长程任务规划能力有限以及资源受限设备上运行效率低下的四大核心挑战，限制了其在实际场景中的实用性和可扩展性。
论文旨在通过提出AppCopilot系统，解决这些问题，实现跨任务、模态、应用和设备的通用性、精准性、长程任务完成能力和高效运行。

## Method

*   **核心思想:** 构建一个多模态、多代理的通用设备端助手，通过端到端的自主流水线设计，形成从数据收集到部署的全栈闭环系统，解决移动代理的四大核心挑战。
*   **数据构建:** 
    *   针对泛化问题，构建中英文通用数据基础，通过系统性应用选择和真实用例任务设计，扩展垂直领域数据多样性，解决中文场景数据稀缺和行为真实性不足的问题。
    *   采用真实用户行为数据收集策略，使用标准化工具提升数据质量。
*   **模型设计:** 
    *   集成多模态基础模型，支持中英文双语能力，通过端到端推理架构减少模块化设计带来的误差累积，提升单步操作精度。
    *   引入OCR和OR结合的区域定位校准机制，增强控件区域识别和语义解析能力，动态校准操作坐标。
*   **推理与控制:** 
    *   结合思维链推理（Chain-of-Thought Reasoning）、层次化任务规划与分解以及多代理协作机制，增强长程任务能力。
    *   支持跨应用任务转换和跨设备协作机制，实现多场景任务协调。
*   **执行优化:** 
    *   通过用户个性化记忆与体验适应、语音交互、功能/工具调用等功能，提升用户体验和任务执行效率。
    *   采用性能分析驱动的优化策略，针对延迟、内存和能耗进行硬件异构优化，确保在资源受限设备上的高效运行。

## Experiment

*   **泛化能力:** AppCopilot在多个基准测试中表现出较强的跨任务和跨应用能力，尤其在中文场景中，相较于基线模型（如UI-TARS、Qwen-VL）有显著提升，验证了数据多样性构建的有效性。
*   **精度提升:** 通过端到端架构和区域定位校准机制，单步操作精度显著提高，特别是在动态界面中，减少了定位误差，与基线模型相比有明显优势。
*   **长程任务:** 在复杂场景任务（如跨应用、跨设备任务）中，完成率高于大多数基线模型，体现了层次化规划和多代理协作的优势。
*   **效率优化:** 通过模型选择、个性化信息检索和历史行为数据复用，AppCopilot在边缘设备上的运行效率得到优化，延迟和能耗均有改善。
*   **实验设置:** 实验涵盖基础能力评估、场景任务评估（通信、交易、娱乐等）以及真实场景测试，设置较为全面，但对极端资源受限设备的测试较少，与闭源模型（如GPT-5）相比仍有一定差距。

## Further Thoughts

AppCopilot的多代理协作机制启发了我，是否可以通过动态角色分配，根据任务复杂度实时调整代理数量和分工，进一步提升长程任务效率？此外，个性化记忆和历史行为复用机制是否可以结合用户行为预测模型，提前缓存常见任务路径，从而进一步降低推理延迟？最后，端到端架构的优势明显，但是否可以通过混合架构（部分模块化+端到端）平衡开发效率和性能优化？