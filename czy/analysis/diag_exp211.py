# -*- coding: utf-8 -*-
"""exp2.11 replay diagnosis: single-leg standing quantification + anti-splay verdict."""
import pandas as pd
import numpy as np

df = pd.read_csv('czy/data/exp2.11/isaac_diag.csv')
seg = (df.step // 500).clip(0, 5).values
cmds = [0.2, 0.4, 0.6, 0.4, 0.2, 0.0]
names = ['hip_p', 'hip_r', 'hip_y', 'knee', 'ank_p', 'ank_r']
# mirrored defaults from config default_joint_angles (L / R)
defL = [0.4, 0.05, -0.31, 0.49, -0.21, 0.0]
defR = [-0.4, -0.05, 0.31, 0.49, 0.21, 0.0]
t = df.time_s.values

print("=" * 76)
print("1) POSTURE: joint means per segment (rad)  [default in header]")
for j in range(6):
    print(f"  {names[j]:7s} dL={defL[j]:+.2f} dR={defR[j]:+.2f} | " +
          " | ".join(f"s{s} L{df[f'dof_pos_{j}'][seg==s].mean():+.2f} R{df[f'dof_pos_{j+6}'][seg==s].mean():+.2f}"
                     for s in [0, 1, 2, 3, 5]))

print("=" * 76)
print("2) HIP_YAW deviation from mirrored default (deg) - anti-splay verdict")
print("    (clamp corridor +-0.85 rad = +-48.7 deg from zero, i.e. dev<31 deg)")
for s in range(6):
    m = seg == s
    devL = np.degrees(df.dof_pos_2[m].mean() - (-0.31))
    devR = np.degrees(df.dof_pos_8[m].mean() - (+0.31))
    mxL = np.degrees((df.dof_pos_2[m] - (-0.31)).abs().max())
    mxR = np.degrees((df.dof_pos_8[m] - (+0.31)).abs().max())
    print(f"  seg{s} cmd={cmds[s]:.1f}: devL={devL:+7.1f} devR={devR:+7.1f}   maxL={mxL:6.1f} maxR={mxR:6.1f}")

print("=" * 76)
print("3) SINGLE-SUPPORT runs (one foot force<1N while other>20N, min 0.3s)")
cl = df.foot_force_l.values > 20.0
cr = df.foot_force_r.values > 20.0
for label, cond in [("L-only (R lifted)", cl & ~cr), ("R-only (L lifted)", cr & ~cl)]:
    runs, st = [], None
    for i, c in enumerate(cond):
        if c and st is None: st = i
        elif not c and st is not None:
            if i - st >= 15: runs.append((st, i))
            st = None
    if st is not None: runs.append((st, len(cond)))
    durs = [(t[b]-t[a]) for a, b in runs]
    tot = sum(durs)
    print(f"  {label}: n={len(runs)}  total={tot:.1f}s ({100*tot/t[-1]:.0f}%)  "
          f"mean={np.mean(durs):.2f}s" if runs else f"  {label}: none")
    for a, b in sorted(runs, key=lambda r: r[1]-r[0], reverse=True)[:6]:
        print(f"      t={t[a]:6.1f}-{t[b]:6.1f}s  dur={t[b]-t[a]:5.2f}s  cmd={cmds[min(int(t[a])//5,5)]:.1f}")

print("=" * 76)
print("4) CONTACT duty per segment (%)")
for s in range(6):
    m = seg == s
    print(f"  seg{s} cmd={cmds[s]:.1f}: L {100*np.mean(cl[m]):3.0f}%  R {100*np.mean(cr[m]):3.0f}%  "
          f"double {100*np.mean(cl[m]&cr[m]):3.0f}%  none {100*np.mean(~cl[m]&~cr[m]):3.0f}%")

print("=" * 76)
print("5) FOOT HEIGHT while 'supporting' (should be ~0 if truly planted)")
for s in [0, 1, 2]:
    m = seg == s
    print(f"  seg{s}: L z={df.foot_z_l[m].mean():.3f} R z={df.foot_z_r[m].mean():.3f} (planted mean)")
    print(f"         L z max={df.foot_z_l[m].max():.3f} R z max={df.foot_z_r[m].max():.3f} (swing apex)")

print("=" * 76)
print("6) GAIT: foot lift events (force drops <1N for >=10 steps = 0.2s)")
for side, force in [("L", df.foot_force_l.values), ("R", df.foot_force_r.values)]:
    air = force < 1.0
    runs, st = [], None
    for i, c in enumerate(air):
        if c and st is None: st = i
        elif not c and st is not None:
            if i - st >= 10: runs.append((st, i))
            st = None
    print(f"  {side}: {len(runs)} swing events, durations={[round(t[b]-t[a],2) for a,b in runs][:20]}")

print("=" * 76)
print("7) LOCOMOTION & HEIGHT")
bh = df.base_height.values
vx = df.base_vel_x.values
for s in range(6):
    m = seg == s
    print(f"  seg{s} cmd={cmds[s]:.1f}: vx={vx[m].mean():+.3f} h={bh[m].mean():.3f} h_std={bh[m].std():.3f}")
print(f"  progress x: {df.base_pos_x.iloc[-1]-df.base_pos_x.iloc[0]:+.2f} m | y: {df.base_pos_y.iloc[-1]-df.base_pos_y.iloc[0]:+.2f} m")
yaw = df.base_yaw.values
wrap = lambda a: (a + np.pi) % (2*np.pi) - np.pi
print(f"  yaw drift: {np.degrees(wrap(yaw[-1]-yaw[0])):+.1f} deg | falls(h<0.42): {100*np.mean(bh<0.42):.0f}%")

print("=" * 76)
print("8) FOOT YAW rel base (deg) - 45deg-hip splay signature")
fyL = wrap(df.foot_yaw_l.values - yaw); fyR = wrap(df.foot_yaw_r.values - yaw)
for s in range(6):
    m = seg == s
    print(f"  seg{s}: L {np.degrees(fyL[m].mean()):+6.1f}  R {np.degrees(fyR[m].mean()):+6.1f} deg")

print("=" * 76)
print("9) ANKLE ROLL (clamped +-0.40 rad) + stance knee during single-support")
for s in [0, 1, 2]:
    m = seg == s
    print(f"  seg{s}: ank_r L={df.dof_pos_5[m].mean():+.2f} R={df.dof_pos_11[m].mean():+.2f} | "
          f"knee L={df.dof_pos_3[m].mean():+.2f} R={df.dof_pos_9[m].mean():+.2f}")

print("=" * 76)
print("10) VELOCITY profile 2s bins (is it trying to walk?)")
for i in range(0, 30, 2):
    m = (t >= i) & (t < i+2)
    print(f"  t={i:2d}-{i+2:2d}s cmd={cmds[min(i//5,5)]:.1f}: vx={vx[m].mean():+.3f} "
          f"h={bh[m].mean():.3f} F_L={df.foot_force_l[m].mean():5.0f} F_R={df.foot_force_r[m].mean():5.0f}")
