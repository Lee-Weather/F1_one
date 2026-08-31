# -*- coding: utf-8 -*-
"""exp2.11: reset-detection + stall-eviction activity check."""
import pandas as pd
import numpy as np

df = pd.read_csv('czy/data/exp2.11/isaac_diag.csv')
t = df.time_s.values
px, py, bh = df.base_pos_x.values, df.base_pos_y.values, df.base_height.values
cmds = [0.2, 0.4, 0.6, 0.4, 0.2, 0.0]

print("== A) teleport detection (|dpos|>0.15m in 1 step = env reset) ==")
d = np.hypot(np.diff(px), np.diff(py))
jumps = np.where(d > 0.15)[0]
print(f"  jumps: {len(jumps)}")
for j in jumps:
    print(f"    t={t[j]:.2f}s pos=({px[j]:.2f},{py[j]:.2f})->({px[j+1]:.2f},{py[j+1]:.2f}) h={bh[j]:.3f}->{bh[j+1]:.3f}")

print("== B) height returns to spawn (~0.69+) ==")
spawn = np.where(bh > 0.69)[0]
if len(spawn):
    # group consecutive
    groups, st = [], spawn[0]
    for a, b in zip(spawn[:-1], spawn[1:]):
        if b - a > 3: groups.append((st, a)); st = b
    groups.append((st, spawn[-1]))
    for a, b in groups:
        print(f"    t={t[a]:.2f}-{t[b]:.2f}s (n={b-a+1})")
else:
    print("    none after t=0")

print("== C) EMA-stall simulation (alpha=0.08, thresh 0.05, 150 steps, cmd>0.3) ==")
vx, vy = df.base_vel_x.values, df.base_vel_y.values
spd = np.hypot(vx, vy)
ema = np.zeros(len(spd)); ema[0] = spd[0]
for i in range(1, len(spd)):
    ema[i] = ema[i-1] + 0.08 * (spd[i] - ema[i-1])
cmd = df.command_x.values
stalled = (np.abs(cmd) > 0.3) & (ema < 0.05)
cnt, fires = 0, []
for i, s in enumerate(stalled):
    cnt = cnt + 1 if s else 0
    if cnt == 150: fires.append(t[i])
print(f"  would-fire timestamps (if counter never reset by env reset): {[round(f,1) for f in fires]}")
print(f"  ema min={ema.min():.3f} max={ema.max():.3f} | frac ema<0.05 & cmd>0.3: {100*np.mean(stalled):.0f}%")
print(f"  NOTE: replay resets would zero ema; if no resets happened these fires are real")

print("== D) vx burst window t=14-16 detail (post-reset transient or real walking?) ==")
m = (t >= 13.5) & (t <= 16.5)
print(f"  vx: {vx[m][::10].round(2)}")
print(f"  h : {bh[m][::10].round(2)}")
print(f"  py: {py[m][::10].round(2)}")

print("== E) cumulative displacement (no-reset proof) ==")
for i in range(0, 3000, 300):
    print(f"    t={t[i]:5.1f}s  pos=({px[i]:+6.2f},{py[i]:+6.2f})  h={bh[i]:.3f}  cmd={cmd[i]:.1f}")
