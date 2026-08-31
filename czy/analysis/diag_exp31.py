# -*- coding: utf-8 -*-
"""exp3.1 replay acceptance: single-leg-standing skill + walk regression.

Checks the pre-registered acceptance table (exp1.md exp3.1 section) from a
--skill_schedule replay CSV (columns: skill_cmd, foot forces / heights / yaws).
Usage: python czy/analysis/diag_exp31.py [path=czy/data/exp3.1/isaac_diag.csv]
"""
import sys
import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else 'czy/data/exp3.1/isaac_diag.csv'
df = pd.read_csv(path)
t = df.time_s.values
PASS, FAIL = [], []

def check(name, val, ok):
    (PASS if ok else FAIL).append(name)
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: {val}")

print("=" * 76)
print("exp3.1 ACCEPTANCE  (skill: L3 0.12m target, schedule 3/10/3/10/3 s)")
print("=" * 76)

skill = df.skill_cmd.values
fl, fr = df.foot_force_l.values, df.foot_force_r.values
contact_l, contact_r = fl > 5, fr > 5
loaded_l, loaded_r = fl > 50, fr > 50

# --- segment the schedule: skill windows are the skill_cmd!=0 runs
runs, st = [], None
for i in range(len(skill)):
    active = skill[i] != 0
    if active and (st is None or skill[i] != skill[st]):
        st = i
    elif not active and st is not None:
        runs.append((st, i, int(skill[st])))
        st = None
if st is not None:
    runs.append((st, len(skill), int(skill[st])))
print(f"skill windows found: {len(runs)} -> " +
      ", ".join(f"{'L' if s==1 else 'R'}@{t[a]:.1f}-{t[b]:.1f}s ({t[b]-t[a]:.1f}s)"
                for a, b, s in runs))

print("\n-- 1) single-support hold per window (target >=10s, redline <5s)")
for a, b, s in runs:
    lift_c, sup_load = (contact_l, loaded_r) if s == 1 else (contact_r, loaded_l)
    clean = (~lift_c[a:b]) & sup_load[a:b]
    # longest clean stretch
    best, cur, cur0 = 0.0, 0, a
    for k, c in enumerate(clean):
        cur = cur + 1 if c else 0
        best = max(best, cur)
    hold = best * 0.01
    check(f"window {'L' if s==1 else 'R'} t={t[a]:.0f}s hold", f"{hold:.1f}s", hold >= 10.0)

print("\n-- 2) lift height in-band ratio (target >=90%, redline <70%)")
for a, b, s in runs:
    h = (df.foot_z_l.values if s == 1 else df.foot_z_r.values)[a:b] - 0.041
    inband = np.mean((h >= 0.06) & (h <= 0.14))
    check(f"window {'L' if s==1 else 'R'} height band", f"{100*inband:.0f}%", inband >= 0.9)

print("\n-- 3) hip_yaw deviation in skill windows (target <=0.25rad=14.3deg)")
names = ['hip_p', 'hip_r', 'hip_y', 'knee', 'ank_p', 'ank_r']
defL = [-0.31, 0.31]
for a, b, s in runs:
    li, si = (2, 8) if s == 1 else (8, 2)
    dev = np.abs(df[f'dof_pos_{li}'][a:b] - defL[0 if s == 1 else 1]).mean()
    dev2 = np.abs(df[f'dof_pos_{si}'][a:b] - defL[1 if s == 1 else 0]).mean()
    check(f"window {'L' if s==1 else 'R'} yaw dev lift/support",
          f"{np.degrees(dev):.1f}/{np.degrees(dev2):.1f} deg",
          dev <= 0.25 and dev2 <= 0.25)

print("\n-- 4) drift during skill windows (target <=0.10m/10s, red 0.3m)")
for a, b, s in runs:
    d = np.hypot(df.base_pos_x.values[b-1] - df.base_pos_x.values[a],
                 df.base_pos_y.values[b-1] - df.base_pos_y.values[a])
    check(f"window {'L' if s==1 else 'R'} drift", f"{d:.2f}m", d <= 0.10)

print("\n-- 5) falls (h<0.42 anytime, target 0, red >3 frames)")
bh = df.base_height.values
check("fall frames", int((bh < 0.42).sum()), (bh < 0.42).sum() == 0)

print("\n-- 6) walk regression: vel tracking in cmd windows (vs exp2.11)")
cmd = df.command_x.values
for seg, target in [(0, 0.2), (1, 0.4), (2, 0.6)]:
    m = (np.abs(cmd - target) < 0.05) & (skill == 0)
    if m.sum() > 50:
        vx = df.base_vel_x.values[m].mean()
        check(f"cmd {target} mean vx", f"{vx:+.3f} m/s", vx >= 0.0)

print("\n" + "=" * 76)
print(f"RESULT: {len(PASS)} passed / {len(FAIL)} failed")
if FAIL:
    print("failed items:")
    for f in FAIL:
        print("  - " + f)
