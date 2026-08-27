# -*- coding: utf-8 -*-
"""exp2.6 回放验收分析 v2：摔倒取证 / 双腿不对称 / 抬腿 / 步频 / yaw / 跟踪。
固定 500 步分段（与 play 日志一致）；复位帧(env 重采样命令的单帧毛刺)单独识别。
用法: python czy/analysis/acceptance_exp26.py [csv_path]
"""
import sys, os
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join("czy", "data", "exp2.6", "isaac_diag.csv")
CONTACT_THR, SEG = 1.0, 500

df = pd.read_csv(CSV)
n = len(df)
DT = float(np.median(np.diff(df["time_s"].values[:100])))
t = df["time_s"].values
h, vx, vy = df["base_height"].values, df["base_vel_x"].values, df["base_vel_y"].values
yaw, px, py = df["base_yaw"].values, df["base_pos_x"].values, df["base_pos_y"].values
fz_l, fz_r = df["foot_z_l"].values, df["foot_z_r"].values
ff_l, ff_r = df["foot_force_l"].values, df["foot_force_r"].values
cyc_min, cyc_max, spd_max = 0.35, 0.70, 0.6
print(f"rows={n} dt={DT:.3f}s dur={n*DT:.1f}s")

# ---------- 1. 复位检测（位置瞬移 >0.5m/帧 = env 重置）----------
dpx, dpy = np.diff(px), np.diff(py)
jumps = np.where((np.abs(dpx) > 0.5) | (np.abs(dpy) > 0.5))[0]
print(f"\n===== 1. 摔倒复位检测: {len(jumps)} 次 =====")
for j in jumps:
    s = max(0, j - 60)  # 复位前 0.6s 取证
    pre_h, pre_v = h[s:j+1], vx[s:j+1]
    pre_yaw = yaw[s:j+1]
    print(f"  t={t[j]:.2f}s 复位 | 前0.6s: h {pre_h[0]:.2f}->{pre_h.min():.2f} | vx峰值{pre_v.max():.2f} "
          f"| yaw漂{np.degrees(pre_yaw[-1]-pre_yaw[0]):+.1f}° | 归属段cmd={[0.2,0.4,0.6,0.4,0.2,0.0][j//SEG]}")
print(f"h_min={h.min():.3f} (h<0.45 帧数={(h<0.45).sum()}; 复位谷底可能短于采样)")
# 复位后段内速度（复位会把进度抹掉）
print(f"注意: 复位即摔倒, 计净前进时需扣除回跳 {sum(dpx[j] for j in jumps):.2f}m")

# ---------- 2. 固定分段指标 ----------
def seg_rows(a, b):
    return slice(a, min(b, n))

print("\n===== 2. 分段: 速度/抬腿/步频/占空比 (固定500步) =====")
hdr = f"{'seg':<4}{'cmd':<5}{'vx均':>6}{'vx峰':>6}{'gapL':>6}{'gapR':>6}{'L/R':>6}{'t3cm':>6}{'步L':>4}{'步R':>4}{'cycL':>6}{'cycR':>6}{'dutyL':>6}{'dutyR':>6}{'yaw°':>7}"
print(hdr)
for si in range(6):
    a, b = si * SEG, (si + 1) * SEG
    sl = seg_rows(a, b)
    cmd = 0.2 * (si + 1) if si < 3 else 0.2 * (6 - si) if si < 5 else 0.0
    cmd = [0.2, 0.4, 0.6, 0.4, 0.2, 0.0][si]
    seg_ffl, seg_ffr = ff_l[sl], ff_r[sl]
    contact_l, contact_r = seg_ffl > CONTACT_THR, seg_ffr > CONTACT_THR
    # 抬腿: 各脚触地高度中位数为基线
    base_l = np.median(fz_l[sl][contact_l]) if contact_l.any() else np.median(fz_l[sl])
    base_r = np.median(fz_r[sl][contact_r]) if contact_r.any() else np.median(fz_r[sl])
    gl = (fz_l[sl] - base_l)[~contact_l] if (~contact_l).any() else np.array([0.0])
    gr = (fz_r[sl] - base_r)[~contact_r] if (~contact_r).any() else np.array([0.0])
    t3 = (np.concatenate([gl, gr]) > 0.03).mean() * 100
    lifto_l = (np.diff(contact_l.astype(int)) == -1).sum()
    lifto_r = (np.diff(contact_r.astype(int)) == -1).sum()
    dur = SEG * DT
    cyc_l = dur / lifto_l if lifto_l else float("nan")
    cyc_r = dur / lifto_r if lifto_r else float("nan")
    lr = np.median(gl) / np.median(gr) if np.median(gr) > 0.001 else float("nan")
    dyaw = np.degrees(yaw[b-1] - yaw[a])
    print(f"{si:<4}{cmd:<5.1f}{vx[sl].mean():>6.2f}{vx[sl].max():>6.2f}"
          f"{np.median(gl)*100:>5.1f}c{np.median(gr)*100:>5.1f}c{lr:>6.2f}{t3:>5.0f}%"
          f"{lifto_l:>4}{lifto_r:>4}{cyc_l:>5.2f}s{cyc_r:>5.2f}s"
          f"{contact_l.mean()*100:>5.0f}%{contact_r.mean()*100:>5.0f}%{dyaw:>+7.1f}")

# ---------- 3. 冲量 / 关节不对称（剔除复位帧±30 帧）----------
mask = np.ones(n, bool)
for j in jumps:
    mask[max(0, j-30):j+30] = False
walk = mask & (vx > 0.1)
print(f"\n===== 3. 不对称量化（剔除复位窗后, 行走帧 {walk.sum()}）=====")
imp_l = np.trapz(ff_l[walk], dx=DT); imp_r = np.trapz(ff_r[walk], dx=DT)
print(f"足底冲量 L={imp_l:.0f} R={imp_r:.0f} N·s  R/L={imp_r/max(imp_l,1e-9):.2f} (exp2.4 修到 0.96)")
pairs = [("hip_pitch", 0, 6), ("hip_roll", 1, 7), ("hip_yaw", 2, 8),
         ("knee_pitch", 3, 9), ("ankle_pitch", 4, 10), ("ankle_roll", 5, 11)]
print(f"{'joint':<12}{'ampL':>7}{'ampR':>7}{'R/L':>6}{'meanL':>7}{'meanR':>7}{'mean差':>7}")
for nm, il, ir in pairs:
    cl, cr = f"dof_pos_{il}", f"dof_pos_{ir}"
    amp_l = np.percentile(df[cl][walk], 95) - np.percentile(df[cl][walk], 5)
    amp_r = np.percentile(df[cr][walk], 95) - np.percentile(df[cr][walk], 5)
    ml, mr = df[cl][walk].mean(), df[cr][walk].mean()
    print(f"{nm:<12}{amp_l:>7.2f}{amp_r:>7.2f}{amp_r/max(amp_l,1e-9):>6.2f}{ml:>7.2f}{mr:>7.2f}{mr-ml:>+7.2f}")

# 支撑占空比全程（不对称主证据）
print(f"全程支撑占空比 L={(ff_l[mask]>CONTACT_THR).mean()*100:.0f}%  R={(ff_r[mask]>CONTACT_THR).mean()*100:.0f}%")
# 单脚支撑比（拐杖步态: 一脚长支撑另一脚摆）
both = ((ff_l > CONTACT_THR) & (ff_r > CONTACT_THR) & mask).mean() * 100
only_l = ((ff_l > CONTACT_THR) & (ff_r <= CONTACT_THR) & mask).mean() * 100
only_r = ((ff_r > CONTACT_THR) & (ff_l <= CONTACT_THR) & mask).mean() * 100
print(f"双支撑={both:.0f}%  仅L支撑={only_l:.0f}%  仅R支撑={only_r:.0f}%")

# ---------- 4. 航向 / 轨迹 ----------
print("\n===== 4. 航向与直线度 (heading off 验证) =====")
print(f"yaw: 初={np.degrees(yaw[0]):+.1f}° 终={np.degrees(yaw[-1]):+.1f}° 峰值|yaw|={np.degrees(np.abs(yaw).max()):.1f}°")
# 每复位段内漂移
edges = [0] + list(jumps + 1) + [n]
for k in range(len(edges)-1):
    a, b = edges[k], edges[k+1]
    if b - a < 50: continue
    dy = np.degrees(yaw[b-1] - yaw[a])
    print(f"  区间 t[{t[a]:.1f},{t[b-1]:.1f}]s yaw漂={dy:+7.1f}° ({dy/((b-a)*DT):+.3f}°/s) y位移={py[b-1]-py[a]:+.2f}m")

# ---------- 5. 净前进（扣复位）----------
net_x = px[-1] - px[0] - sum(dpx[j] for j in jumps)
print(f"\n净前进(扣复位跳变)={net_x:.2f}m / 30s (目标≥4.5, 红线3)")

# ---------- 6. 力矩 ----------
T = df[[f"dof_torque_{i}" for i in range(12)]][walk].values
print(f"力矩 RMS={np.sqrt((T**2).mean()):.1f} 峰值={np.abs(T).max():.0f} Nm")
zc = sum((np.diff(np.sign(df[c][walk].values)) != 0).sum() for c in ["dof_torque_4","dof_torque_5","dof_torque_10","dof_torque_11"])
print(f"踝抖零交叉={zc} ({zc/4:.0f}/关节/{walk.sum()*DT:.0f}s 行走帧)")
