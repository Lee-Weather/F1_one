# Exp1 实验笔记

## 实验索引

| 编号 | 日期 | 摘要 | 状态 | Task ID | GM账号 | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| exp0 | 2026-08-10 | 无参考轨迹 X1 平地行走 3000 轮基线训练，训练/回放达标，CSV 分析显示步态为小碎步，需下一轮改进 | 已测试 | TASK_20260810_071 | jevid17601@barumart.com | model_3000.pt |
| exp0.1 | 2026-08-10 | 0.7s 周期/步长奖励，训练在 2600 轮因账号余额终止；play 周期 0.154s，未达标 | 失败 | TASK_20260810_105 | jevid17601@barumart.com | model_2600.pt |

## 实验 exp0：无参考轨迹 3000 轮基线

### 1. 上一实验结果与教训

> 本轮为 exp 系列首个实验，无上一轮数据。
> 基于上游仓库默认配置直接开训，作为基线。

### 2. 本轮修改目标

- 目标1：验证无参考轨迹的 X1 平地行走策略在 3000 轮训练后的收敛情况。
- 目标2：训练完成后运行 play_gm，输出 `isaac_diag.csv` 与回放 MP4。
- 验收标准：Mean reward >= 25，Mean episode length >= 900，play 平均前进速度接近 0.5 m/s。

### 3. 修改内容

### 修改一：训练最大轮数设为 3000

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| max_iterations | 6000 | 3000 | 缩短基线训练轮数 |

**理由**：控制训练时长与成本，验证 3000 轮是否足够收敛。

### 修改二：play_gm 增加 CSV 记录与上传

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| play_gm.py | 仅保存 model_diag.pt | 增加 isaac_diag.csv，并打包 model_isaac_csv.pt | 回放诊断数据可下载 |

**理由**：需要拿到 play 输出的 CSV 数据。

### 4. 修改文件

- `humanoid/envs/x1/x1_dh_stand_config.py`：`max_iterations` 6000 -> 3000。
- `humanoid/scripts/play_gm.py`：新增 `save_diag_csv`、`package_csv_as_pt`，CSV 包写入 SDK 扫描的 `gm_play` 目录。
- `.gitignore`：忽略 `api_key.json`、`.obsidian/` 与 `czy/data/`。
- `skills/` -> `czy/skills/`：技能目录迁移。
- `czy/skills/gm-cli/SKILL.md`：补充 gm 获取 play CSV/MP4 的流程。

### 5. 训练参数

| 参数 | 值 |
| --- | --- |
| 训练方式 | 从零 |
| GM账号 | jevid17601@barumart.com |
| max_iterations | 3000 |
| save_interval | 100 |
| num_envs | 4096 |
| seed | 5 |
| learning_rate | 1e-3 |
| 算力 | 1x4090D 24G, ESKU000001 |
| 镜像 | BJX00000001, V000021 |
| 代码仓库 | https://github.com/Lee-Weather/F1_one.git, main |
| 启动命令 | `gm-run F1_one/humanoid/scripts/train.py --task=x1_dh_stand --headless --max_iterations=3000` |

### 6. 预期与验收

| 指标 | 上一轮 | 本轮目标 | 异常信号 |
| --- | --- | --- | --- |
| Mean reward | - | >= 25 | < 15 |
| Mean episode length | - | >= 900 | < 700 |
| track_lin_vel_xy_exp | - | > 1.0 | < 0.5 |
| track_ang_vel_z_exp | - | > 0.5 | < 0.2 |
| termination_penalty | - | 接近 0 | 明显负值 |

### 7. 实验结果

> 训练任务：TASK_20260810_071，2026-08-10 12:45:40 完成。
> 最终 checkpoint：model_3000.pt
> 回放任务：TASK_20260810_095，2026-08-10 13:16:20 完成。

#### 最终结果（iter 2999）

| 指标 | 上一轮 | 目标 | 实测 | 判定 |
| --- | --- | --- | --- | --- |
| Mean reward | - | >= 25 | 120.78 | ✅ |
| Mean episode length | - | >= 900 | 2265.51 | ✅ |

#### 训练趋势

| iter | Mean reward | Mean episode length |
| --- | --- | --- |
| 308 | 60.74 | 2030.27 |
| 2999 | 120.78 | 2265.51 |

#### 各奖励项最终值

| 奖励项 | 权重 | 最终值 | 说明 |
| --- | --- | --- | --- |
| tracking_lin_vel | 1.8 | 1.2128 | 线速度跟踪 |
| tracking_ang_vel | 1.1 | 0.6495 | 角速度跟踪 |
| default_joint_pos | 2.0 | 1.2717 | 默认关节姿态 |
| orientation | 1.0 | 0.5949 | 躯干姿态 |
| feet_air_time | 2.0 | 0.0179 | 摆腿离地 |
| termination | -2.0 | -0.0000 | 终止惩罚 |
| max_command_x | - | 1.5000 | 指令课程上限 |

#### Play 回放结果

| 指标 | 实测 |
| --- | --- |
| 平均前进速度 | 0.588 m/s（指令 0.5 m/s） |
| 平均躯干高度 | 0.569 m |
| 回放步数 | 2000 |
| CSV | `czy/data/exp0/isaac_diag.csv` |
| MP4 | `czy/data/exp0/play_output.mp4` |

#### Play 数据诊断（isaac_diag.csv）

| 指标 | 实测 |
| --- | --- |
| 每条腿步频 | 约 8.0 步/s |
| 单腿步周期 | 约 0.122 s |
| 单腿摆腿时间 | 约 0.049 s |
| 估算步长 | 约 0.073 m |
| 抬脚高度 | 约 0.016~0.020 m |
| 双支撑占比 | 约 21% |
| 腾空占比 | 约 0.45% |
| 诊断图 | `czy/data/exp0/analysis_small_steps.png` |

**结论**：⚠️部分达标（数值指标达标，但步态质量不理想：策略收敛为高频小碎步）。

**根因分析**：
- 速度跟踪奖励只约束整体速度，未约束步长/步频，策略选择“高频小步”这一更稳的局部最优。
- `feet_air_time`、`feet_height`、`feet_clearance` 权重不足以逼出大幅摆腿。
- 单步摆腿仅约 0.05s，抬脚约 0.02m，步长约 0.073m，表现为小碎步。

**下一轮方向**：
- 增加最小摆腿时间/步周期约束，抑制高频小步。
- 增加步长或步频惩罚奖励，促使策略拉开步幅。
- 提高抬脚高度/摆动高度相关奖励权重。
- 视情况加入简单步态相位或参考轨迹约束。

## 实验 exp0.1：0.7s 步周期奖励调整

### 1. 上一实验结果与教训

> 数据：exp0 play `isaac_diag.csv`
> - 每条腿步频约 8.0 步/s，单腿周期约 0.122s，估算步长约 0.073m，抬脚高度约 0.016~0.020m。
> - 速度跟踪达标（0.588 m/s），但步态为高频小碎步。

**核心教训**：
- 仅靠速度奖励会被策略用“高频小步”满足；
- 需要显式约束步周期、摆腿时间和步长。

### 2. 本轮修改目标

- 目标1：单腿步周期接近 0.7s。
- 目标2：摆腿时间 >= 0.10s，估算步长 >= 0.15m。
- 验收标准：每条腿步频 <= 3.0 步/s，步周期接近 0.7s。

### 3. 修改内容

### 修改三：新增 step_cycle 奖励

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| cycle_time_target | 无 | 0.7s | 周期目标 |
| cycle_time_sigma | 无 | 0.1 | 高斯奖励宽度 |
| step_cycle | 无 | 2.0 | 奖励权重 |

**理由**：直接对单腿步周期做高斯奖励，压制 0.12s 小碎步。

### 修改四：新增 stride_length 奖励

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| stride_length_min | 无 | 0.20m | 最小步长 |
| stride_length_max | 无 | 0.35m | 满分步长 |
| stride_length | 无 | 1.5 | 奖励权重 |

**理由**：让策略把摆动脚真正向前跨出。

### 修改五：摆腿时间改目标区间

| 参数 | 旧值 | 新值 | 说明 |
| --- | --- | --- | --- |
| swing_time_target | 无 | 0.25s | 目标摆腿时间 |
| min_swing_time | 无 | 0.15s | 最小摆腿时间 |
| feet_air_time | 2.0 | 4.0 | 提高权重 |
| target_feet_height | 0.03m | 0.05m | 提高抬脚 |
| feet_height | 0.5 | 1.0 | 提高权重 |
| feet_clearance | 0.5 | 1.0 | 提高权重 |

**理由**：避免短促摆腿也能拿到空气时间奖励。

### 修改六：动作平滑惩罚

| 参数 | 旧值 | 新值 |
| --- | --- | --- |
| action_smoothness | -0.002 | -0.01 |
| dof_acc | -1e-7 | -3e-7 |

**理由**：抑制高频关节抖动。

### 4. 修改文件

- `humanoid/envs/x1/x1_dh_stand_config.py`
- `humanoid/envs/x1/x1_dh_stand_env.py`

### 5. 训练参数

| 参数 | 值 |
| --- | --- |
| 训练方式 | 从零 |
| GM账号 | jevid17601@barumart.com |
| max_iterations | 3000（实际到 2600 被终止） |
| save_interval | 100 |
| num_envs | 4096 |
| seed | 5 |
| learning_rate | 1e-3 |
| 算力 | 1x4090D 24G, ESKU000001 |
| 镜像 | BJX00000001, V000021 |
| 代码仓库 | https://github.com/Lee-Weather/F1_one.git, main |
| 启动命令 | `gm-run F1_one/humanoid/scripts/train.py --task=x1_dh_stand --headless --max_iterations=3000` |

### 6. 预期与验收

| 指标 | exp0 | 目标 | 异常信号 |
| --- | --- | --- | --- |
| 每条腿步频 | 8.0 | <= 3.0 | > 5.0 |
| 单腿步周期 | 0.122s | ~0.7s | < 0.3s |
| 估算步长 | 0.073m | >= 0.15m | < 0.10m |
| 摆腿时间 | 0.049s | >= 0.10s | < 0.07s |

### 7. 实验结果

> 训练任务：TASK_20260810_105，2026-08-10 15:39:49 因账号余额不足被终止。
> 最终 checkpoint：model_2600.pt
> 回放任务：TASK_20260810_142（账号 memokaf419@barumart.com）。

#### 最终结果（iter 2600）

| 指标 | exp0 | 目标 | 实测 | 判定 |
| --- | --- | --- | --- | --- |
| 单腿步周期 | 0.122s | ~0.7s | 0.154s | ❌ |
| 每条腿步频 | 8.0 | <= 3.0 | 6.3 | ❌ |
| 估算步长 | 0.073m | >= 0.15m | 0.095m | ❌ |
| 摆腿时间 | 0.049s | >= 0.10s | 0.06s | ❌ |
| 抬脚高度 | 0.011m | >= 0.03m | 0.016m | ❌ |

#### 训练趋势

| iter | Mean reward | Mean episode length |
| --- | --- | --- |
| 644 | 85.58 | 2154.69 |
| 1403 | 113.27 | 2279.40 |
| 2120 | 121.46 | 2343.97 |
| 2630 | 115.67 | 2234.37 |

#### 新奖励实际值

| 奖励项 | 权重 | 最终值 |
| --- | --- | --- |
| step_cycle | 2.0 | 0.0000 |
| stride_length | 1.5 | 0.0069 |
| feet_air_time | 4.0 | 0.0082 |
| action_smoothness | -0.01 | -0.3897 |
| dof_acc | -3e-7 | -0.2041 |

**结论**：❌未达标（周期 0.154s 相对目标 0.7s 仍很远，但相对 exp0 有改善：步频 8.0 -> 6.3，步长 0.073 -> 0.095m）。

**根因分析**：
- `step_cycle` / `stride_length` 奖励最终值接近 0，说明奖励信号太稀疏或太弱，策略几乎没有被引导到 0.7s 周期；
- 训练在 2600 轮因账号余额终止，未跑满 3000 轮；
- 仅加奖励而观测里没有相位/周期提示，策略不容易自己稳定地“按 0.7s 节奏迈步”。

**下一轮方向**：
- 提高 `step_cycle` / `stride_length` 权重，或从较短周期（如 0.3s）做课程逐步过渡到 0.7s；
- 在观测中增加每只脚的相位/时钟信号；
- 继续用剩余账号从 model_2600 续训到 3000，或直接重训；
- 若奖励方案仍无效，再考虑加简单步态参考。