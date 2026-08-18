# exp1.4 acceptance analysis: straightness, falls, survival, tracking
import csv
import numpy as np

ROWS = list(csv.DictReader(open(r'e:\X1\F1_one\F1_one\czy\data\exp1.4\isaac_diag.csv', encoding='utf-8')))
f = lambda k: np.array([float(r[k]) for r in ROWS])
cmd, vx, vy, vyaw = f('command_x'), f('base_vel_x'), f('base_vel_y'), f('base_vel_yaw')
h, px, py, yaw = f('base_height'), f('base_pos_x'), f('base_pos_y'), f('base_yaw')
N = len(ROWS); dt = 0.02
CMDS = [0.2, 0.4, 0.6, 0.4, 0.2, 0.0]
SEGS = [i * 500 for i in range(6)]

print('== 1. FALL EVENTS (h<0.35) ==')
lowh = np.where(h < 0.35)[0]
if len(lowh):
    for g in np.split(lowh, np.where(np.diff(lowh) > 25)[0] + 1):
        print(f"  t={g[0]*dt:.1f}..{g[-1]*dt:.1f}s h_min={h[g].min():.3f} vyaw={np.abs(vyaw[g]).max():.2f}")
else:
    print('  NONE - zero falls across 60s')

print('== 2. SURVIVAL ==')
print(f"  h_min overall={h.min():.3f} (fall threshold 0.35); steps with h>0.5: {(h>0.5).mean()*100:.1f}%")

print('== 3. PER-SEGMENT TRACKING ==')
for i, (c, s) in enumerate(zip(CMDS, SEGS)):
    e = s + 500
    print(f"  seg{i} cmd={c:.1f}: vx={vx[s:e].mean():.3f} ({vx[s:e].mean()/c*100 if c else 0:.0f}%)  vy={vy[s:e].mean():+.3f}  h={h[s:e].mean():.3f}")

print('== 4. TRANSITION DYNAMICS ==')
for idx, c0, c1 in [(500,0.2,0.4),(1000,0.4,0.6),(1500,0.6,0.4),(2000,0.4,0.2),(2500,0.2,0.0)]:
    seg = vx[idx:idx+500]
    over = max(0.0, seg.max() - c1) if c1 > c0 else 0.0
    hw = h[idx:idx+200]; fall = 'FALL' if hw.min() < 0.35 else 'ok'
    print(f"  {c0}->{c1}: v@2s={seg[:100].mean():.3f} v@4s={seg[100:200].mean():.3f} segmax={seg.max():.3f} overshoot={over:.2f} [{fall}]")

print('== 5. STRAIGHTNESS ==')
dx, dy = px[-1]-px[0], py[-1]-py[0]
print(f"  end displacement dx={dx:.2f} dy={dy:.2f}; heading={np.degrees(np.arctan2(dy,dx)):.1f} deg")
for i, (c, s) in enumerate(zip(CMDS[:-1], SEGS[:-1])):
    e = s + 500
    ddx, ddy = px[e]-px[s], py[e]-py[s]
    if ddx**2+ddy**2 > 0.01:
        print(f"  seg{i}: heading={np.degrees(np.arctan2(ddy,ddx)):+.1f} deg  lateral_end={ddy:+.3f}m")
print(f"  yaw range: {np.degrees(yaw).min():+.1f}..{np.degrees(yaw).max():+.1f} deg")
