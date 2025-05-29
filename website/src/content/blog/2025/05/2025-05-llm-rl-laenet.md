---
title: "Large Language Model-enhanced Reinforcement Learning for Low-Altitude Economy Networking"
pubDatetime: 2025-05-27T11:25:42+00:00
slug: "2025-05-llm-rl-laenet"
type: "arxiv"
id: "2505.21045"
score: 0.5916472877062774
author: "grok-3-latest"
authors: ["Lingyi Cai", "Ruichen Zhang", "Changyuan Zhao", "Yu Zhang", "Jiawen Kang", "Dusit Niyato", "Tao Jiang", "Xuemin Shen"]
tags: ["LLM", "Reinforcement Learning", "Reward Design", "Decision Making", "Multimodal Processing"]
institution: ["Huazhong University of Science and Technology", "Nanyang Technological University", "Guangdong University of Technology", "University of Waterloo"]
description: "本文提出了一种LLM增强的RL框架，通过LLM作为信息处理器、奖励设计者、决策者和生成器的多角色功能，提升了低空经济网络中强化学习的决策灵活性和性能，并在能耗优化案例中验证了其有效性。"
---

> **Summary:** 本文提出了一种LLM增强的RL框架，通过LLM作为信息处理器、奖励设计者、决策者和生成器的多角色功能，提升了低空经济网络中强化学习的决策灵活性和性能，并在能耗优化案例中验证了其有效性。 

> **Keywords:** LLM, Reinforcement Learning, Reward Design, Decision Making, Multimodal Processing

**Authors:** Lingyi Cai, Ruichen Zhang, Changyuan Zhao, Yu Zhang, Jiawen Kang, Dusit Niyato, Tao Jiang, Xuemin Shen

**Institution(s):** Huazhong University of Science and Technology, Nanyang Technological University, Guangdong University of Technology, University of Waterloo


## Problem Background

低空经济网络（LAENet）旨在通过部署无人机等空中平台支持1000米以下空域的多样化飞行应用（如智能交通、灾难响应），但面临实时决策、环境不确定性和资源限制等挑战。
传统的强化学习（RL）虽能应对动态环境下的自主决策，却受限于泛化能力不足、奖励函数设计复杂和模型稳定性差等问题，论文因此提出利用大型语言模型（LLM）的生成、上下文理解和结构化推理能力，增强RL以解决LAENet中的关键问题。

## Method

*   **框架概述:** 提出了一种LLM增强的RL框架，将LLM集成到RL的多个阶段，充分发挥其高层次认知能力与RL策略学习的互补优势，具体通过以下四个角色实现：
*   **信息处理器:** LLM处理复杂多模态数据（如图像、文本指令），提取关键特征并生成状态表示，减轻RL代理的计算负担。例如，将天气变化或地面指令转化为RL状态空间中的特征向量（如‘visibility=low’），提升环境适应性。
*   **奖励设计者:** LLM利用预训练知识和推理能力，动态生成适应性奖励函数，避免手动设计的复杂性和静态局限。例如，根据任务目标（如‘最小化能耗’）生成可执行的奖励函数代码，并通过训练反馈迭代优化，确保奖励与任务目标对齐。
*   **决策者:** LLM通过上下文理解和结构化推理，生成高质量动作候选或专家策略，缩小RL动作搜索空间，提高决策效率。例如，在无人机轨迹优化中，基于当前状态（如位置、能量）生成动作建议（如‘向北低速移动’），辅助RL代理选择最优动作。
*   **生成器:** LLM预测未来状态和奖励序列，支持基于模型的RL学习，减少对真实环境交互的依赖；同时生成自然语言解释RL策略，提升决策透明度和可信度。例如，生成模拟轨迹数据或解释无人机移动决策的原因（如‘为降低能耗选择东移’）。
*   **工作流程:** 以无人机辅助物联网数据收集为例，LLM在状态感知、动作选择、奖励评估和策略更新各阶段嵌入RL循环，提升系统在动态环境中的智能性和适应性。

## Experiment

*   **案例研究:** 聚焦LLM作为奖励设计者在LAENet中的应用，场景为无人机辅助物联网数据收集，目标是最小化总能耗，采用DDPG和TD3两种深度强化学习算法进行验证。
*   **有效性:** 实验结果表明，LLM（GPT-4o）设计的奖励函数显著优于手动设计，TD3算法最终能耗降低7.2%，DDPG也有类似改进，尤其在小数据包（2.0 Mbits）场景下能耗降低达6.2%。这是因为LLM通过推理引入了更丰富的奖励因素（如无人机位置对能耗的影响），优化了轨迹选择和通信开销。
*   **设置合理性:** 实验设置包括不同数据包大小（2.0-2.8 Mbits）对能耗的影响，模拟环境（300m×300m区域，10个物联网终端，Rician衰落信道）贴近实际应用，训练参数（如学习率10^-4和3×10^-4，批次大小64，折扣因子0.99）符合常规实践，验证了方法的鲁棒性。
*   **局限性:** 实验仅验证了奖励设计功能，未全面测试LLM在信息处理、决策和生成方面的效果；实验规模较小（单一场景，终端数量有限），可能无法完全反映LAENet复杂性；LLM生成奖励的随机性和幻觉问题虽通过多候选生成和约束评估缓解，但计算开销未详细分析。

## Further Thoughts

LLM的多角色应用启发了我思考其在动态调整RL超参数（如学习率、折扣因子）中的潜力，以适应不同任务需求；此外，LLM在多模态数据处理中的能力不仅适用于LAENet，也可能扩展至自动驾驶或机器人控制等领域，通过统一处理视觉、语言和传感器数据提升泛化性；论文提到的多代理RL与多个LLM协作的未来方向也令人期待，或将为无人机集群等分布式系统的协作优化带来新突破。