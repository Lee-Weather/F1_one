# 实验修改记录（基于源码 origin）

## v3 — 关节物理阻尼（joint damping）改造

相对源码（origin）的差异。本版本聚焦于**仿真器物理关节阻尼**的域随机化方式，使其使用配置中显式定义的基准值，而不再依赖 URDF 中的原始阻尼值。

> 说明：这里改的是 Isaac Gym 的 DOF 物理阻尼 `dof_props["damping"]`，与 PD 控制器的 `d_gains`（`control.damping` 字典）是两套独立机制，后者未改动。

---

### 1. 配置改动 `humanoid/envs/x1/x1_dh_stand_config.py`

`DH_X1Cfg.domain_rand` 中的 joint damping 部分。

源码（origin）：

```python
randomize_joint_damping = True
randomize_joint_damping_each_joint = False
joint_damping_range = [0.3, 1.5]
joint_1_damping_range = [0.3, 1.5]
joint_2_damping_range = [0.3, 1.5]
joint_3_damping_range = [0.3, 1.5]
joint_4_damping_range = [0.9, 1.5]
joint_5_damping_range = [0.9, 1.5]
joint_6_damping_range = [0.3, 1.5]
joint_7_damping_range = [0.3, 1.5]
joint_8_damping_range = [0.3, 1.5]
joint_9_damping_range = [0.9, 1.5]
joint_10_damping_range = [0.9, 1.5]
```

v3：

```python
randomize_joint_damping = True
randomize_joint_damping_each_joint = False
initial_joint_damping = [2.0, 2.0, 2.0, 1.5, 0.5, 0.5, 2.0, 2.0, 2.0, 1.5, 0.5, 0.5]
joint_damping_range = [0.8, 1.2]       # global fallback (each_joint=False uses this)
joint_1_damping_range = [0.8, 1.2]
joint_2_damping_range = [0.8, 1.2]
joint_3_damping_range = [0.8, 1.2]
joint_4_damping_range = [0.8, 1.2]
joint_5_damping_range = [0.8, 1.2]
joint_6_damping_range = [0.8, 1.2]
joint_7_damping_range = [0.8, 1.2]
joint_8_damping_range = [0.8, 1.2]
joint_9_damping_range = [0.8, 1.2]
joint_10_damping_range = [0.8, 1.2]
joint_11_damping_range = [0.8, 1.2]
joint_12_damping_range = [0.8, 1.2]
```

改动点：
- 新增 `initial_joint_damping`：12 个关节（左腿6 + 右腿6，顺序 hip_pitch / hip_roll / hip_yaw / knee_pitch / ankle_pitch / ankle_roll）的物理阻尼**绝对基准值**。
- 全局 `joint_damping_range` 由 `[0.3, 1.5]` 收窄为 `[0.8, 1.2]`（multiplier，±20%）。
- 各关节单独范围统一为 `[0.8, 1.2]`，并补齐 `joint_11`、`joint_12`（源码只到 joint_10）。

---

### 2. 逻辑改动 `humanoid/envs/base/legged_robot.py`

函数 `_refresh_actor_dof_props`，在每个 DOF 循环内、阻尼随机化之前，新增用配置基准值覆盖 URDF 阻尼的逻辑。

源码（origin）：

```python
if self.cfg.domain_rand.randomize_joint_damping:
    if self.cfg.domain_rand.randomize_joint_damping_each_joint:
        dof_props["damping"][i] *= self.joint_damping_coeffs[env_id, i]
    else:
        dof_props["damping"][i] *= self.joint_damping_coeffs[env_id, 0]
```

v3：

```python
# Override the URDF damping with the config-defined base value
# so that initial_joint_damping takes effect (not the URDF value).
initial_damping = getattr(self.cfg.domain_rand, 'initial_joint_damping', None)
if initial_damping is not None:
    dof_props["damping"][i] = initial_damping[i]

if self.cfg.domain_rand.randomize_joint_damping:
    if self.cfg.domain_rand.randomize_joint_damping_each_joint:
        dof_props["damping"][i] *= self.joint_damping_coeffs[env_id, i]
    else:
        dof_props["damping"][i] *= self.joint_damping_coeffs[env_id, 0]
```

改动点：
- 物理阻尼基准从 **URDF 原始值** 改为 **`initial_joint_damping[i]`**（绝对覆盖）。
- 随机系数仍按原逻辑乘在基准值之上。
- 用 `getattr(..., None)` 保护：未定义该字段的任务/config 自动退回源码行为（URDF × 系数），不受影响。

---

### 3. 实际生效效果（当前配置 `each_joint = False`）

随机系数为每个环境采样一个全局标量（所有关节共用），范围 `[0.8, 1.2]`。

| 关节（DOF 顺序） | 基准值 | 训练时范围（× 0.8~1.2） |
|---|---|---|
| joint 1-3（hip pitch/roll/yaw） | 2.0 | 1.6 ~ 2.4 |
| joint 4（knee pitch） | 1.5 | 1.2 ~ 1.8 |
| joint 5-6（ankle pitch/roll） | 0.5 | 0.4 ~ 0.6 |
| joint 7-9（hip pitch/roll/yaw） | 2.0 | 1.6 ~ 2.4 |
| joint 10（knee pitch） | 1.5 | 1.2 ~ 1.8 |
| joint 11-12（ankle pitch/roll） | 0.5 | 0.4 ~ 0.6 |

- 训练时（`randomize_joint_damping = True`）：基准值 × 全局随机系数。
- play 时（`play.py` 中 `randomize_joint_damping = False`）：直接使用这 12 个基准值，完全替代 URDF。

---

### 4. 注意事项

- `initial_joint_damping` 按 `dof_props` 的索引 `i` 直接对应，即 Isaac Gym 加载 asset 时的 DOF 顺序，需确认与预期（左腿6 + 右腿6）一致。
- `joint_*_damping_range`（逐关节范围）仅在 `randomize_joint_damping_each_joint = True` 时才生效；当前为 `False`，实际只用全局 `joint_damping_range`。
