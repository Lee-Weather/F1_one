# -*- coding: utf-8 -*-
# exp1.1 失败模式分析: czy/data/exp1.1/isaac_diag.csv
# 1) 摔倒时刻定位与摔倒模式判定  2) 各段前向跟踪误差分解
# 3) 起步段左偏瞬态  4) 步态周期/步长 vs 理论自适应周期  5) 扭矩RMS与足底冲击
import csv
import numpy as np

CSV_PATH = r"e:\X1\F1_one\F1_one\czy\data\exp1.1\isaac_diag.csv"
CONTACT_N = 1.0          # 足底力 > 1N 视为支撑
JUMP_M = 1.0             # 单步位置突变阈值(reset)
DT_CTRL = 0.02           # 任务标称控制周期
SEGMENTS = [(0, 500, 0.2), (500, 1000, 0.4), (1000, 1500, 0.6),
            (1500, 2000, 0.4), (2000, 2500, 0.2), (2500, 3000, 0.0)]
STEADY_TAIL = 250        # 每段取后 250 步作稳态
SEG_NAMES = ["seg0", "seg1", "seg2", "seg3", "seg4", "seg5"]


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    keys = list(rows[0].keys())
    data = {k: np.array([float(row[k]) for row in rows]) for k in keys}
    return data


def wrap_deg(a):
    return float(np.degrees(np.arctan2(np.sin(a), np.cos(a))))


def stance_edges(force):
    """支撑掩码的上升沿索引(忽略首样本)"""
    mask = force > CONTACT_N
    edges = np.where(mask & ~np.roll(mask, 1))[0]
    return edges[edges > 0]


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


d = load(CSV_PATH)
N = len(d["step"])
t = d["time_s"]
dt_rec = float(np.median(np.diff(t)))          # 记录时间列的实际间隔
print(f"数据: {N} 行, time_s 记录间隔 dt_rec={dt_rec:.4f}s (总时长 {t[-1]:.2f}s)")
print(f"标称控制周期 DT_CTRL={DT_CTRL}s. 后文周期同时给 dt_rec / DT_CTRL 两种解释.")

px, py = d["base_pos_x"], d["base_pos_y"]
h = d["base_height"]
vx, vy, wz = d["base_vel_x"], d["base_vel_y"], d["base_vel_yaw"]
yaw = d["base_yaw"]
fl, fr = d["foot_force_l"], d["foot_force_r"]
fzl, fzr = d["foot_z_l"], d["foot_z_r"]
TQ = np.stack([d[f"dof_torque_{i}"] for i in range(12)], axis=1)   # (N,12)

# ---------------------------------------------------------------- 1. 摔倒定位
section("1. 摔倒/reset 精确定位")
jump = np.hypot(np.diff(px), np.diff(py))
falls = np.where(jump > JUMP_M)[0] + 1        # 突变发生行
if len(falls) == 0:
    print("未检测到位置突变(无 reset)")
else:
    for f_idx in falls:
        print(f"\n--- reset @ step {f_idx}, time_s={t[f_idx]:.2f}, "
              f"pos ({px[f_idx-1]:.3f},{py[f_idx-1]:.3f}) -> ({px[f_idx]:.3f},{py[f_idx]:.3f}), "
              f"单步跳变 {jump[f_idx-1]:.2f} m")
        w0 = max(0, f_idx - 200)              # 摔倒前 200 步窗口
        W = slice(w0, f_idx)
        hw, flw, frw = h[W], fl[W], fr[W]
        vxw, vyw = vx[W], vy[W]
        tqw = TQ[W]
        tq_peak = np.abs(tqw).max()
        tq_peak_j = int(np.abs(tqw).max(axis=0).argmax())
        tq_rms_w = float(np.sqrt((tqw ** 2).mean()))
        h0, hmin, hend = hw[0], hw.min(), hw[-1]
        # 双脚离地起点(最后连续 50 步内力都 < 1N 视为腾空/失支撑)
        no_sup = np.where((flw < CONTACT_N) & (frw < CONTACT_N))[0]
        print(f"  窗口[{w0},{f_idx}) 200步: base_height {h0:.3f} -> min {hmin:.3f} -> 末值 {hend:.3f} (跌落 {h0-hend:.3f} m)")
        print(f"  foot_force_l: peak {flw.max():.1f}N 末值 {flw[-1]:.1f}N | foot_force_r: peak {frw.max():.1f}N 末值 {frw[-1]:.1f}N")
        last_sup = w0 + (no_sup[0] if len(no_sup) else -1)
        print(f"  双脚同时无支撑首现: step {last_sup} (距摔点 {f_idx-last_sup} 步)" if len(no_sup)
              else "  窗口内双脚始终至少一脚有支撑力")
        print(f"  base_vel_x: [{vxw.min():+.3f},{vxw.max():+.3f}] 末段10步均值 {vxw[-10:].mean():+.3f}")
        print(f"  base_vel_y: [{vyw.min():+.3f},{vyw.max():+.3f}] 末段10步均值 {vyw[-10:].mean():+.3f}")
        print(f"  累计侧移 dy={py[f_idx-1]-py[w0]:+.3f} m, 航向变化 {wrap_deg(yaw[f_idx-1]-yaw[w0]):+.1f} deg")
        print(f"  dof_torque: 窗口峰值 |{tq_peak:.1f}| Nm @joint{tq_peak_j}, 窗口RMS(12关节) {tq_rms_w:.2f} Nm")
        print(f"  foot_z_l [{fzl[W].min():.3f},{fzl[W].max():.3f}]  foot_z_r [{fzr[W].min():.3f},{fzr[W].max():.3f}]")
        print(f"  reset后首步: h={h[f_idx]:.3f}, vx={vx[f_idx]:+.3f}, vy={vy[f_idx]:+.3f}, fl={fl[f_idx]:.1f}, fr={fr[f_idx]:.1f}")
        print("  -- 摔倒前最后 100 步采样(每10步): step  h     fl     fr     vx     vy    tqlmax")
        for k in range(f_idx - 100, f_idx, 10):
            print(f"     {k:5d} {h[k]:.3f} {fl[k]:6.1f} {fr[k]:6.1f} {vx[k]:+.3f} {vy[k]:+.3f} {np.abs(TQ[k]).max():6.1f}")
        print(f"     {f_idx-1:5d} {h[f_idx-1]:.3f} {fl[f_idx-1]:6.1f} {fr[f_idx-1]:6.1f} {vx[f_idx-1]:+.3f} {vy[f_idx-1]:+.3f} {np.abs(TQ[f_idx-1]).max():6.1f}")

# 摔点之后段内统计需要拆分: 若 seg 内有 reset, 拆成 pre/post
fall_in = {}
for f_idx in falls:
    for si, (a, b, c) in enumerate(SEGMENTS):
        if a <= f_idx < b:
            fall_in.setdefault(si, []).append(f_idx)

# ---------------------------------------------------------------- 2. 各段跟踪误差分解
section("2. 各段稳态前向跟踪误差分解 (每段后250步)")
print(f"{'段':<5}{'cmd':>5}{'vx实测':>8}{'误差%':>7}{'vy均值':>8}{'vy/vx%':>7}{'yaw率':>7}{'段内侧移m':>9}{'段内航向Δ°':>10}")
for si, (a, b, c) in enumerate(SEGMENTS):
    s = slice(b - STEADY_TAIL, b)
    mvx = vx[s].mean()
    err = (c - mvx) / c * 100 if c > 1e-6 else 0.0
    mvy = vy[s].mean()
    mwz = np.degrees(wz[s].mean())
    dy_seg = py[b - 1] - py[a]
    dyaw = wrap_deg(yaw[b - 1] - yaw[a])
    name = SEG_NAMES[si] + ("*" if si in fall_in else "")
    print(f"{name:<5}{c:>5.2f}{mvx:>8.3f}{err:>7.1f}{mvy:>8.3f}"
          f"{abs(mvy)/max(mvx,1e-6)*100:>7.1f}{mwz:>7.2f}{dy_seg:>+9.3f}{dyaw:>+10.1f}")
print("(* = 段内含摔倒reset, 稳态窗口可能受 reset 后恢复期影响)")

if 3 in fall_in:  # seg3 拆分细看
    f_idx = fall_in[3][0]
    for nm, a2, b2 in [("seg3摔前", 1500, f_idx), ("seg3摔后", f_idx + 50, 2000)]:
        if b2 - a2 < 60:
            continue
        s = slice(a2, b2)
        print(f"  {nm}[{a2}:{b2}]: vx均值 {vx[s].mean():.3f}, vy均值 {vy[s].mean():+.3f}, "
              f"高h均值 {h[s].mean():.3f}, 高h末50步均值 {h[a2:b2][-50:].mean():.3f}")

# ---------------------------------------------------------------- 3. 起步段左偏瞬态
section("3. seg0 起步左偏瞬态 (前300步, 每25步采样)")
print(f"{'step':>5}{'t(dt.01)':>9}{'t(dt.02)':>9}{'pos_y':>8}{'yaw°':>7}{'vy':>8}{'vx':>7}{'fl':>6}{'fr':>6}")
for k in range(0, 301, 25):
    print(f"{k:>5}{k*dt_rec:>9.2f}{k*DT_CTRL:>9.2f}{py[k]:>+8.3f}{np.degrees(yaw[k]):>7.2f}"
          f"{vy[k]:>+8.3f}{vx[k]:>7.3f}{fl[k]:>6.1f}{fr[k]:>6.1f}")
k_ = np.arange(min(400, N))
i_peak_vy = int(np.argmax(vy[:400]))
i_peak_py = int(np.argmax(np.abs(py[:400])))
conv = np.where((np.arange(50, 400)[np.abs(vy[50:400]) < 0.02]))[0]
t_conv = int(conv[0]) if len(conv) else -1
print(f"\n  vy 峰值 {vy[:400].max():+.3f} @step {i_peak_vy} (t={i_peak_vy*dt_rec:.2f}s记录)")
print(f"  |pos_y| 峰值 {np.abs(py[:400]).max():.3f} m @step {i_peak_py}")
print(f"  vy 首次回落到 |vy|<0.02: step {t_conv} (t={t_conv*dt_rec:.2f}s记录)" if t_conv > 0 else "  400步内 vy 未收敛到 0.02 以内")
s0 = slice(0, 500)
vy0_mean = vy[s0].mean()
print(f"  seg0 全段: vy 均值 {vy0_mean:+.3f}, 段末侧移 {py[499]:+.3f} m, 段末航向 {np.degrees(yaw[499]):+.2f}°, "
      f"段末 yaw率 {np.degrees(wz[450:500].mean()):+.2f}°/s")

# ---------------------------------------------------------------- 4. 步态周期与步长
section("4. 步态周期与步长 (foot_force 上升沿)")
print(f"{'段':<5}{'cmd':>5}{'步数/周期L':>9}{'周期@dt01':>9}{'周期@dt02':>9}{'理论周期':>8}{'比(02/理论)':>10}"
      f"{'步长实测m':>9}{'步长@cmd m':>10}{'节奏Hz@02':>9}")
for si, (a, b, c) in enumerate(SEGMENTS):
    if si in fall_in:  # 段内含reset则只统计摔前
        f_idx = fall_in[si][0]
        b = f_idx
    if b - a < 100:
        print(f"{SEG_NAMES[si]:<5}{c:>5.2f}   样本不足")
        continue
    el = stance_edges(fl[a:b])
    er = stance_edges(fr[a:b])
    if len(el) < 3:
        print(f"{SEG_NAMES[si]:<5}{c:>5.2f}   支撑沿过少(L:{len(el)} R:{len(er)}) ~原地站立")
        continue
    steps_cyc = float(np.median(np.diff(el)))
    per01 = steps_cyc * dt_rec
    per02 = steps_cyc * DT_CTRL
    theo = min(0.35 + c / 0.6 * 0.35, 0.7)
    mvx = vx[b - STEADY_TAIL:b].mean()
    stride_meas = per02 * mvx
    stride_cmd = per02 * c
    print(f"{SEG_NAMES[si]:<5}{c:>5.2f}{steps_cyc:>9.1f}{per01:>9.3f}{per02:>9.3f}{theo:>8.3f}"
          f"{per02/theo:>10.2f}{stride_meas:>9.3f}{stride_cmd:>10.3f}{1.0/per02:>9.2f}")
print("说明: 周期@dt01=按记录time_s, 周期@dt02=按标称控制50Hz; 理论=0.35+cmd/0.6*0.35 (上限0.7)")
print("      步长=周期×对应速度; 节奏Hz=1/周期@dt02")

# ---------------------------------------------------------------- 5. 扭矩与冲击
section("5. 各段扭矩RMS/峰值与足底冲击")
print(f"{'段':<5}{'cmd':>5}{'TQ_RMS':>8}{'TQ_peak':>9}{'peak关节':>8}{'F_peak':>8}{'F_中位峰':>9}{'支撑占空比L':>10}{'h均值':>7}")
for si, (a, b, c) in enumerate(SEGMENTS):
    if si in fall_in:
        f_idx = fall_in[si][0]
        b = f_idx
    if b - a < 100:
        print(f"{SEG_NAMES[si]:<5}{c:>5.2f}   样本不足")
        continue
    tq = TQ[a:b]
    rms = float(np.sqrt((tq ** 2).mean()))
    pk = float(np.abs(tq).max())
    pj = int(np.abs(tq).max(axis=0).argmax())
    fpk = max(fl[a:b].max(), fr[a:b].max())
    # 每次支撑期的峰值力中位数
    def stance_peaks(f):
        m = f > CONTACT_N
        peaks, s = [], None
        for i, v in enumerate(m):
            if v and s is None:
                s = i
            elif not v and s is not None:
                peaks.append(f[s:i].max())
                s = None
        if s is not None:
            peaks.append(f[s:].max())
        return peaks
    pl = stance_peaks(fl[a:b])
    fmed = float(np.median(pl)) if pl else 0.0
    duty = float((fl[a:b] > CONTACT_N).mean())
    print(f"{SEG_NAMES[si]:<5}{c:>5.2f}{rms:>8.2f}{pk:>9.1f}{('j'+str(pj)):>8}{fpk:>8.1f}{fmed:>9.1f}{duty:>10.2f}{h[a:b].mean():>7.3f}")
# 摔倒窗口扭矩对比
for f_idx in falls:
    w = TQ[max(0, f_idx - 200):f_idx]
    print(f"  摔倒前200步: TQ_RMS {np.sqrt((w**2).mean()):.2f} Nm, 峰值 {np.abs(w).max():.1f} Nm (对比所在段均值)")

# ---------------------------------------------------------------- 全局
section("全局")
print(f"base_height 全程: min {h.min():.3f} @step {int(h.argmin())}, 均值 {h.mean():.3f}, 末值 {h[-1]:.3f}")
lowh = np.where(h < 0.5)[0]
print(f"  h<0.5 的步数: {len(lowh)}" + (f", 首个 step {lowh[0]}" if len(lowh) else ""))
print(f"净位移: x {px[-1]-px[0]:+.2f} m, y {py[-1]-py[0]:+.2f} m, 末航向 {np.degrees(yaw[-1]):+.1f}°")
