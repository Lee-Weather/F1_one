---
name: "sim2sim-test"
description: "训练完成后在 MuJoCo 中执行 sim2sim 测试。Invoke when training completes and user wants to run sim2sim validation."
---

# Sim2Sim 测试流程

训练完成后，将训练模型下载到本地并在 MuJoCo 中执行 sim2sim 验证测试。

## 1. 前置条件

- 训练任务已完成（`gm task status` 状态为 done）
- MuJoCo MJCF 模型已清洗并可用（路径见下文"环境配置"）
- conda 环境 `F1` 可用

## 2. 目录结构约定

```
~/czy/roboparty/lab_test/czy/data/models/x1_29_models/
  └── {YYYYMMDD}_{实验名}/        ← 实验独立文件夹（如 20260731_x1_29_exp1.1）
      ├── model_3000.pt           ← 训练 checkpoint
      ├── policy_3000.pt          ← JIT 转换后的策略
      ├── joint_positions.png     ← sim2sim 关节位置图
      ├── base_velocities.png     ← sim2sim 基座速度图
      └── isaac_diag_*.csv        ← sim2sim 诊断数据

~/czy/exp_data/                    ← 项目外视频目录
  └── {YYYYMMDD_HHMMSS}_{实验名}_sim2sim.mp4
```

### 命名规则

| 项目 | 规则 | 示例 |
|------|------|------|
| 实验独立文件夹 | `{日期}_{实验名}` | `20260731_x1_29_exp1.1` |
| 视频文件名 | `{YYYYMMDD_HHMMSS}_{实验名}_sim2sim.mp4` | `20260731_140000_x1_29_exp1.1_sim2sim.mp4` |

## 3. 执行步骤

### 步骤 1：下载训练模型

**前提**：任务必须**正常完成**（状态 5）。训练中/被停止的任务 GM 不会上传模型（模型列表恒为空，拿不到）。

```bash
# 列出任务的全部模型（响应字段为 data.rows，不是 data.modelList）
gm task model list --task-id {TASK_ID} --limit 20

# 取 model_{N}_deploy.pt 的 policUrlDown 链接，用 curl 下载
curl -sL -o model_3000_deploy.pt "<policUrlDown>"
```

- **首选下载 `model_{N}_deploy.pt`**：训练结束时 train.py 自动从训练对象导出的 deploy JIT（结构 `_TorchPolicyExporter(actor, normalizer)`），下载后 `torch.jit.load` **直接用于 sim2sim，无需转换**。
- 如需 checkpoint（续训/分析/convert），下载 `model_{N}.pt`。
- 创建实验独立文件夹（如不存在），命名格式：`{YYYYMMDD}_{实验名}`

> **不要使用 `gm task data get` 下载模型**——那是图表数据接口，返回的是训练曲线数据，不是模型文件。

### 步骤 2：转换模型为 JIT

> **仅当只有 checkpoint（`model_{N}.pt`）时才需要转换**；若已下载 `model_{N}_deploy.pt`（deploy JIT）可跳过此步。

```bash
cd /home/robot/czy/roboparty/lab_test

# 输入: model_3000.pt (checkpoint)
# 输出: policy_3000.pt (JIT policy)
/home/robot/Anaconda/envs/F1/bin/python robolab/scripts/mujoco/convert_x1_29_checkpoint.py \
    --checkpoint czy/data/models/x1_29_models/{YYYYMMDD}_{实验名}/model_3000.pt \
    --output czy/data/models/x1_29_models/{YYYYMMDD}_{实验名}/policy_3000.pt
```

### 步骤 3：运行 sim2sim（无头模式）

当前阶段：**仅测试原地站立**（vx=0, vy=0, dyaw=0）

```bash
# 进入实验独立文件夹（输出文件存放位置）
cd /home/robot/czy/roboparty/lab_test/czy/data/models/x1_29_models/{YYYYMMDD}_{实验名}/

# 确保视频输出目录存在
mkdir -p ~/czy/exp_data

# 记录时间戳和实验名用于视频命名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="{实验名}"

# 运行 sim2sim
MUJOCO_GL=egl \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
X1_29_MJCF=/tmp/x1_29_mjcf_clean/mjmodel_x1_29dof_perfect_mirrored_sim_flat.xml \
/home/robot/Anaconda/envs/F1/bin/python \
    /home/robot/czy/roboparty/lab_test/robolab/scripts/mujoco/sim2sim_x1_29.py \
    --load_model policy_3000.pt \
    --headless

# 移动视频到项目外目录并重命名
mv simulation.mp4 ~/czy/exp_data/${TIMESTAMP}_${EXP_NAME}_sim2sim.mp4
```

### 步骤 4：验证输出文件

实验独立文件夹中**必须包含**：

| 文件 | 说明 |
|------|------|
| `model_{N}.pt` | 训练 checkpoint |
| `policy_{N}.pt` | JIT 转换后的策略 |
| `joint_positions.png` | 29 关节指令 vs 实际位置图 |
| `base_velocities.png` | 基座线速度/角速度图 |
| `isaac_diag_*.csv` | 168 列诊断数据（含接触力、关节力矩等） |

项目外目录中：

| 文件 | 说明 |
|------|------|
| `{时间戳}_{实验名}_sim2sim.mp4` | 仿真视频 |

## 4. 环境配置（一次性）

### MJCF 模型清洗

如果 `/tmp/x1_29_mjcf_clean/` 不存在，需先清洗：

```bash
# 复制原始 MJCF
cp czy/data/x1_29/mjcf/mjmodel_x1_29dof_perfect_mirrored_sim_flat.xml /tmp/x1_29_mjcf_clean/

# 移除 MuJoCo 不支持的属性
cd /tmp/x1_29_mjcf_clean
sed -i 's/ content_type="model\/stl"//g' mjmodel_x1_29dof_perfect_mirrored_sim_flat.xml
sed -i 's/ actuatorfrcrange="[^"]*"//g' mjmodel_x1_29dof_perfect_mirrored_sim_flat.xml

# 设置 meshdir 为绝对路径
sed -i 's|meshdir="[^"]*"|meshdir="/tmp/x1_29_mjcf_clean/meshes/"|' mjmodel_x1_29dof_perfect_mirrored_sim_flat.xml
# 添加 autolimits 和 offscreen 支持
sed -i 's|<compiler |<compiler autolimits="true" |' mjmodel_x1_29dof_perfect_mirrored_sim_flat.xml
sed -i 's|<visual>|<visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>|' mjmodel_x1_29dof_perfect_mirrored_sim_flat.xml

# 创建 mesh 软链接
ln -sf /home/robot/czy/roboparty/lab_test/czy/data/x1_29/meshes /tmp/x1_29_mjcf_clean/meshes

# 设置 offscreen 帧缓冲区大小
sed -i 's|<option |<option offwidth="1920" offheight="1080" |' mjmodel_x1_29dof_perfect_mirrored_sim_flat.xml
```

### 关键路径

| 项目 | 路径 |
|------|------|
| sim2sim 脚本 | `/home/robot/czy/roboparty/lab_test/robolab/scripts/mujoco/sim2sim_x1_29.py` |
| checkpoint 转换脚本 | `/home/robot/czy/roboparty/lab_test/robolab/scripts/mujoco/convert_x1_29_checkpoint.py` |
| MJCF 清洗模型 | `/tmp/x1_29_mjcf_clean/mjmodel_x1_29dof_perfect_mirrored_sim_flat.xml` |
| Python 环境 | `/home/robot/Anaconda/envs/F1/bin/python` |
| 模型存储根目录 | `/home/robot/czy/roboparty/lab_test/czy/data/models/x1_29_models/` |
| 视频输出目录 | `~/czy/exp_data/` |

## 5. 诊断数据说明

CSV 文件（168 列）包含：

| 类别 | 列数 | 内容 |
|------|------|------|
| 时间 | 1 | timestamp_ns |
| 指令 | 3 | cmd_linear_x/y, cmd_angular_z |
| 姿态 | 7 | euler_xyz, quat_wxyz |
| 速度 | 6 | base_ang_vel_xyz, base_lin_vel_xyz |
| 接触 | 6 | left/right_contact, foot_force_z, foot_force_mag |
| 关节×29 | 145 | action, pos, vel, effort, pos_des |

可用于评估：
- **稳定性**：基座姿态角度是否在 ±5° 以内
- **接触状态**：双支撑率是否 >70%
- **跟踪精度**：关节 |pos - pos_des| 平均误差是否 <3°
- **力矩饱和**：是否有关节持续满载

## 6. 注意事项

- 原地站立测试（全0指令）是训练中 20% env 的工况，策略理应能应对
- 如果机器人翻滚，首先排查观测构造和 PD 增益是否与训练一致
- 仿真时长可在 `sim2sim_x1_29.py` 的 `sim_duration` 中修改（默认 20s）
- 所有 sim2sim 运行都应使用 `--headless` 和 `MUJOCO_GL=egl`
