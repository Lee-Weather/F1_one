# -*- coding: utf-8 -*-
"""exp2.9 replay diagnosis: splay posture quantification."""
import pandas as pd
import numpy as np

df = pd.read_csv('czy/data/exp2.9/isaac_diag.csv')
t = df['time_s'].values
N = len(df)
seg = np.minimum(df['step'].values // 500, 5)
cmds = [0.2, 0.4, 0.6, 0.4, 0.2, 0.0]

D = df  # alias
hyL = D['dof_pos_2'].values   # left hip_yaw, default -0.31
hyR = D['dof_pos_8'].values   # right hip_yaw, default +0.31
hrL = D['dof_pos_1'].values   # left hip_roll, default +0.05
hrR = D['dof_pos_7'].values   # right hip_roll
knL = D['dof_pos_3'].values   # left knee, default 0.49
knR = D['dof_pos_9'].values
devL = hyL - (-0.31)
devR = hyR - (+0.31)
bh = D['base_height'].values
yaw = D['base_yaw'].values

print("=" * 70)
print("1) FALL CHECK (base_height < 0.40 sustained)")
fell = bh < 0.40
falls, in_fall = [], False
for i, f in enumerate(fell):
    if f and not in_fall:
        st, in_fall = i, True
    elif not f and in_fall:
        in_fall = False
        if i - st > 25: falls.append((t[st], round(bh[st],2), round(bh[st-10],2) if st>10 else -1))
if in_fall: falls.append((t[st], round(bh[st],2), -1))
print(f"   height min={bh.min():.3f}  final={bh[-1]:.3f}  falls>0.5s: {len(falls)}")
for f in falls: print(f"   t={f[0]:.2f}s h={f[1]}")

print("=" * 70)
print("2) HIP YAW DEVIATION (deg from mirrored default, sign corrected)")
def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi
print(f"   devL: mean={np.degrees(np.mean(devL)):+.1f}  |max|={np.degrees(np.max(np.abs(devL))):.1f}  P10={np.degrees(np.percentile(devL,10)):+.1f} P90={np.degrees(np.percentile(devL,90)):+.1f}")
print(f"   devR: mean={np.degrees(np.mean(devR)):+.1f}  |max|={np.degrees(np.max(np.abs(devR))):.1f}  P10={np.degrees(np.percentile(devR,10)):+.1f} P90={np.degrees(np.percentile(devR,90)):+.1f}")
print(f"   L at-limit(|dev|>80deg) frame%: {100*np.mean(np.abs(devL)>np.radians(80)):.0f}%")
print(f"   R at-limit(|dev|>80deg) frame%: {100*np.mean(np.abs(devR)>np.radians(80)):.0f}%")
print(f"   devL first 20s mean: {np.degrees(np.mean(devL[:1000])):+.1f} | 40-60s mean: {np.degrees(np.mean(devL[2000:])):+.1f}")
print(f"   devR first 20s mean: {np.degrees(np.mean(devR[:1000])):+.1f} | 40-60s mean: {np.degrees(np.mean(devR[2000:])):+.1f}")

print("=" * 70)
print("3) FOOT ORIENTATION rel base (foot_yaw - base_yaw)")
fyL = wrap(D['foot_yaw_l'].values - yaw)
fyR = wrap(D['foot_yaw_r'].values - yaw)
print(f"   foot_yaw_L: mean={np.degrees(np.mean(fyL)):+.1f}  P10={np.degrees(np.percentile(fyL,10)):+.1f} P90={np.degrees(np.percentile(fyL,90)):+.1f}")
print(f"   foot_yaw_R: mean={np.degrees(np.mean(fyR)):+.1f}  P10={np.degrees(np.percentile(fyR,10)):+.1f} P90={np.degrees(np.percentile(fyR,90)):+.1f}")
print(f"   |foot yaw| > 45deg frames: L {100*np.mean(np.abs(fyL)>np.radians(45)):.0f}% / R {100*np.mean(np.abs(fyR)>np.radians(45)):.0f}%")
print(f"   spread (R-L): mean={np.degrees(np.mean(fyR-fyL)):+.1f}")

print("=" * 70)
print("4) OTHER LEG JOINTS (mean)")
print(f"   hip_roll  L {np.degrees(np.mean(hrL)):+.1f} / R {np.degrees(np.mean(hrR)):+.1f} deg (default +2.9)")
print(f"   knee      L {np.mean(knL):+.2f} / R {np.mean(knR):+.2f} rad (default 0.49)")
arL = D['dof_pos_5'].values; arR = D['dof_pos_11'].values
print(f"   ankle_roll L {np.degrees(np.mean(arL)):+.1f} / R {np.degrees(np.mean(arR)):+.1f} deg (default 0, limit -0.64)")

print("=" * 70)
print("5) SPEED TRACKING per segment (50 steps warmup skipped)")
for s in range(6):
    m = seg == s
    v = D['base_vel_x'].values[m][50:]
    print(f"   seg{s} cmd={cmds[s]:.1f}: real={v.mean():.3f} ({100*v.mean()/cmds[s] if cmds[s]>0 else 0:.0f}%)  vx_peak={v.max():.2f}")

print("=" * 70)
print("6) YAW / PATH")
print(f"   yaw(0)={np.degrees(yaw[0]):+.1f}  yaw(end)={np.degrees(yaw[-1]):+.1f}  drift={np.degrees(wrap(yaw[-1]-yaw[0])):+.1f} deg over 60s")
print(f"   lateral |y| drift: {abs(D['base_pos_y'].values[-1]-D['base_pos_y'].values[0]):.2f} m")
print(f"   forward progress: {D['base_pos_x'].values[-1]-D['base_pos_x'].values[0]:.2f} m")

print("=" * 70)
print("7) CONTACT / DUTY / IMPULSE (force>1N)")
cl = D['foot_force_l'].values > 1.0
cr = D['foot_force_r'].values > 1.0
walk = seg.isin([1,2,3]).values if hasattr(seg,'isin') else np.isin(seg,[1,2,3])
print(f"   duty walk(10-40s): L {100*np.mean(cl[walk]):.0f}% / R {100*np.mean(cr[walk]):.0f}%  (diff {abs(100*np.mean(cl[walk])-100*np.mean(cr[walk])):.0f}pp)")
print(f"   only-L {100*np.mean(cl[walk] & ~cr[walk]):.0f}%  only-R {100*np.mean(cr[walk] & ~cl[walk]):.0f}%  double {100*np.mean(cl[walk] & cr[walk]):.0f}%")
impL = np.sum(D['foot_force_l'].values[walk]); impR = np.sum(D['foot_force_r'].values[walk])
print(f"   impulse ratio R/L: {impR/impL:.2f}")

print("=" * 70)
print("8) FOOT LIFT (swing apex, walk segs 1-3)")
fzl = D['foot_z_l'].values; fzr = D['foot_z_r'].values
for s in [0,1,2]:
    m = (seg == s).copy()
    if s == 0: m[:50] = False
    swL = m & (~cl) & cr; swR = m & (~cr) & cl
    gl = 1000*np.percentile(fzl[swL], 50) if swL.sum() > 5 else float('nan')
    gl9 = 1000*np.percentile(fzl[swL], 90) if swL.sum() > 5 else float('nan')
    gr = 1000*np.percentile(fzr[swR], 50) if swR.sum() > 5 else float('nan')
    gr9 = 1000*np.percentile(fzr[swR], 90) if swR.sum() > 5 else float('nan')
    print(f"   seg{s} cmd={cmds[s]:.1f}: gap L P50={gl:.1f}cm P90={gl9:.1f}cm | R P50={gr:.1f}cm P90={gr9:.1f}cm  (swing frames L {swL.sum()} R {swR.sum()})")
print(f"   foot_z sanity: L min={1000*fzl.min():.1f} P50={1000*np.percentile(fzl,50):.1f} max={1000*fzl.max():.1f} cm")

print("=" * 70)
print("9) HIP_YAW TIME SERIES (10s bins, deg)")
for i in range(0, 60, 10):
    m = (t >= i) & (t < i+10)
    print(f"   t={i:2d}-{i+10:2d}s  cmd={cmds[min(i//10,5)]:.1f}  devL={np.degrees(np.mean(devL[m])):+6.1f}  devR={np.degrees(np.mean(devR[m])):+6.1f}  fYawL={np.degrees(np.mean(fyL[m])):+6.1f}  fYawR={np.degrees(np.mean(fyR[m])):+6.1f}")
