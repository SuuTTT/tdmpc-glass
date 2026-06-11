# DR2 (GPT) — 图/结构化世界模型在强化学习中的进展

*Saved 2026-06-11. External deep-research report #2 (GPT). Verbatim archive (Chinese).*

近年来研究者提出多种**图结构/对象中心**世界模型提高复杂环境控制性能。节点=场景中独立实体（物体/代理），通过 GNN 或 Transformer 建模实体间交互。Slot 模型用无/带监督编码器（Slot Attention、SLATE、SAVi、DINOSAUR）从图像/点云分离对象特征，再用 GNN/Transformer 预测动态。

代表工作：
- **ObjectZero (2601.06604)**: 预训练 SLATE/DINOSAUR 生成对象槽 + 全连通 GNN 动态 + MCTS；CausalWorld 多物体到达、Robosuite 块提升。块提升任务略优于 DreamerV3/ROCA，接近 EfficientZero。
- **FOCUS (2307.02427)**: 对象中心潜在动态 + 对象潜在熵探索奖励；ManiSkill2/Robosuite。Lift Cube 收敛更快，Stack/Turn 比 DreamerV2 高 10%/35%；稀疏奖励下显著提升探索。
- **SOLD (2410.08822)**: 首个像素端无监督对象中心模型（SAVi + Slot Aggregation Transformer 预测 reward/value）；需要物体关联推理的视觉操纵基准上显著优于 DreamerV3 和 TD-MPC2；注意力图可解释。
- **STICA (2511.14262)**: 端到端对象中心 Transformer（WM+策略+值都基于对象 token）；SafetyGym 3D 障碍 + 2D/3D 视觉 RL；超越先前 SOTA，因果注意可视化。
- **SSWM (2402.03326)**: Slot Attention + GNN，Spriteworld 无监督对象发现+动态预测；多步预测超基线但长程仍难。
- **OC-STORM (2501.16443)**: 少量标注帧+预训练分割加入 STORM；Atari/Hollow Knight；显著提高样本效率。
- **CEE-US (NeurIPS 2022)**: 多物体自由交互，GNN 集成（节点=物体+机械臂），GNN 不确定性=内在奖励；比 MLP 更快产生丰富交互数据，提升零样本泛化。
- **RD-GNN (2209.11943)**: 点云场景图嵌入动态，多步物体重排规划。

总体：图/对象中心模型在**需推理多实体及其交互**的任务上突出（FOCUS/SOLD 多物体操纵优于 Dreamer；ObjectZero 块提升超 DreamerV3；STICA 3D 障碍领先）。**单物体/简单动力学场景，单一潜在模型往往足够**（ObjectZero 到达任务与 EZ-V2 持平）。

## 结构熵与图抽象在世界模型中的应用
**未发现已有文献将结构熵作为可微损失直接施加于学习到的世界模型图结构。** 图聚类/池化（DiffPool/MinCutPool/SAGPool）存在于 GNN 领域但几乎无 MBRL 世界模型应用。GraphIB 多用于图分类。社区检测/层次聚合（Louvain）一般用于分析图数据而非在线学习目标。选项/技能发现有基于拉普拉斯的早期工作但处理已知状态图。**在学习世界模型的潜在/动力学图上用结构熵最小化或层次聚合损失尚属新颖，无直接先例。**

## 图结构在控制任务中的潜在优势
多主体/多物体相互作用时结构化模型有优势（FOCUS/SOLD 多物体；CEE-US 更快交互数据；STICA 3D）。Iso-Dream 分离可控代理 vs 不可控其他，多车道场景提升。相反证据：单物体/简单场景单体已足够。**真正增长空间：高度组合的物体排列、新物体组合的公理化推理、多智能体交互**——单潜在无法有效捕捉的结构化情境。传统 Mujoco 简单场景改进有限。

## 推荐架构与实验方案
- 节点=独立可感知物体/机器人部件（实例分割/Slot Attention/预训练 SLATE/DINOSAUR/SAVi）。
- 边=完全连接（任意交互）或空间邻近/接触屏蔽。
- 模型= GNN（明确 pairwise）或 Transformer（全局依赖/长时记忆），或混合。
- 结构熵/聚合损失：DiffPool/MinCutPool 逐层聚合超节点，或可微熵正则最小化结构熵增大群集——**慎防聚合过度丢失重要差异**。
- 规划：想象回放（Dreamer）或显式规划（MCTS，ObjectZero）。
- 基准：多物体操纵（MuJoCo/IsaacGym 摆放、Robosuite/ManiSkill 堆叠装配、长时序多目标）、CausalWorld。
- 对照：DreamerV3、TD-MPC2、ROCA/OCRL；无世界模型 SAC/PPO+CNN。
- 陷阱：对象发现不确定性（数量变化/遮挡/身份漂移）；聚合过度混合或过分分离；图模型计算代价（GNN 二次复杂度），收敛敏感；结构损失可能被网络绕过；少物体场景图≈单体浪费容量。
- 风险检测：训练 GNN WM 看预测误差/探索是否优于基线；看潜在向量是否自然分离对象；若结构化无额外信息（不提高线性预测精度）→ 方向难奏效。

## 结论
多物体/多智能体场景引入图结构有潜力但非无风险。需对象关联推理的操纵任务图模型提高效率/泛化；简单任务单体足够。**结构熵作为显式抽象目标无先例（新颖性）但缺直接经验证据。** 务必与强基线对比 + 探索任务评估群集效果。若早期试验图模型与基线无显著差异（或复杂性大增收益微弱）→ 单体已隐式学到相关结构。先小规模快速验证（简单多物体上比对单体 vs 图：预测误差、探索质量、价值解码），无改进再考虑投入。

Refs: 2601.06604, 2307.02427, 2501.16443, 2402.03326, 2209.11943, 2410.08822, 2511.14262.
