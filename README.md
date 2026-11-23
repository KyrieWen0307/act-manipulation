# ACT 双臂操作实战（ALOHA Transfer Cube + LeRobot）

基于 Datawhale《从零到一搭建一台具身智能机器人》（`datawhalechina/dive-into-embodied-ai`）的 ACT（Action Chunking with Transformers）系统实战仓库。

本项目拒绝“下载即运行”的快餐复现，采用 **“原理吃透 → 挖空手写核心模块 + 离线单元测试 → GPU 断点续训基线 + 破坏性消融实验”** 的工程路线。

## 硬件与算力分工
- **第 1 ~ 3 章（手写核心模块 + 离线单测 + 1-batch 过拟合）**：Win11 轻薄本（20 核 CPU，复用本地 `PyTorch CPU` + 项目 `.venv`）
- **第 4 ~ 5 章（GPU 基线训练 + MuJoCo 20 回合评测 + 破坏性消融）**：宿舍 RTX 4060 8GB 游戏本（支持断电断点续训）或 Google Colab GPU

## 5 章学习与实验路线图

| 章节 | 目录 | 核心任务 | 状态 |
| :--- | :--- | :--- | :--- |
| 第 1 章 | `chapter01_cvae/` | 原理过关 + 手动探索 Hugging Face 数据集 + 手写 `CVAEStyleEncoder`（重参数化、KL 散度、Padding Mask 隔离）与 `pytest` 单测 | `[ ] 进行中` |
| 第 2 章 | `chapter02_mini_act/` | 手写 1D/2D 正弦位置编码 + Mini-ACT 主干（视觉 Token + Policy Encoder + Action Queries Cross-Attention Decoder + 推理强制 $z=0$） | `[ ] 待开始` |
| 第 3 章 | `chapter03_loss_ensemble/` | 手写 Masked L1 + $\beta \cdot \text{KL}$ 联合损失、CPU 1-batch 过拟合冒烟与 Temporal Ensembling 平滑对比 | `[ ] 待开始` |
| 第 4 章 | `chapter04_lerobot_baseline/` | LeRobot 数据审计 + 断点续训跑通 `lerobot/aloha_sim_transfer_cube_human` 50k 步基线 + 20 回合 MuJoCo 闭环评测与 GIF 留档 | `[ ] 待开始` |
| 第 5 章 | `chapter05_ablation/` | 三组破坏性消融（推理策略消融 / `chunk_size` 消融 / `use_vae` 消融）与失败案例诊断复盘 | `[ ] 待开始` |

## 参考基线与数据集
- 理论参考：`datawhalechina/dive-into-embodied-ai` → `docs/foundations/vla/6.act_action_chunking.md`
- 实战参考：`datawhalechina/dive-into-embodied-ai` → `docs/practices/vla/act/index.md`
- 数据集：`lerobot/aloha_sim_transfer_cube_human`（50 episodes, 50 Hz, 20,000 frames, `observation.images.top` `[3, 480, 640]`, `observation.state` `[14]`, `action` `[14]`）
- 上游参考指标（50k steps, batch_size=8, 20 回合 MuJoCo 评测）：成功率 50%（10/20），平均最大奖励 2.8 / 4.0，平均累计奖励 161.6
