# exp1.6 acceptance: lift >=8cm all segs, seg3/4 drift <=8deg, red lines
import csv
import numpy as np

ROWS = list(csv.DictReader(open(r'e:\X1\F1_one\F1_one\czy\data\exp1.6\isaac_diag.csv', encoding='utf-8')))
f = lambda k: np.array([float(r[k]) for r in ROWS])
cmd, vx, vy, vyaw = f('command_x'), f('base_vel_x'), f('base_vel_y'), f('base_vel_yaw')
h, px, py, yaw = f('base_height'), f('base_pos_x'), f('base_pos_y'), f('base_yaw')
fzl, fzr, fl, fr = f('foot_z_l'), f('foot_z_r'), f('foot_force_l'), f('foot_force_r')
tau = np.array([[float(r[f'dof_torque_{j}']) for j in range(12)] for r in ROWS])
SEGS = [(i*500, c) for i, c in enumerate([0.2,0.4,0.6,0.4,0.2,0.0])]
onL, onR = fl > 20, fr > 20
dt = 0.02

print('== 1. LIFT per segment (target >=8cm everywhere) ==')
for i, (s, c) in enumerate(SEGS[:5]):
    e = s+500
    out = []
    for side, fz, on in [('L', fzl, onL), ('R', fzr, onR)]:
        sw = ~on[s:e]
        edges = np.where(np.diff(sw.astype(int))==1)[0]+1
        peaks = [fz[s+edges[k]:s+edges[k+1]].max() for k in range(len(edges)-1) if edges[k+1]-edges[k]>3]
        out.append(f'{side}={np.mean(peaks)*100:.1f}cm(max {max(peaks)*100:.1f})' if peaks else f'{side}=n/a')
    print(f'  seg{i} cmd={c:.1f}: ' + ' '.join(out))

print('== 2. HEADING per segment (target |deg|<=8 for seg3/4) ==')
for i, (s, c) in enumerate(SEGS):
    e = min(s+500, len(px))
    ddx, ddy = px[e]-px[s], py[e]-py[s]
    if ddx**2+ddy**2 > 0.01:
        print(f'  seg{i} cmd={c:.1f}: heading={np.degrees(np.arctan2(ddy,ddx)):+.1f} yaw={np.degrees(yaw[s]):+.1f}->{np.degrees(yaw[e-1]):+.1f}')

print('== 3. RED LINES ==')
lowh = np.where(h<0.35)[0]
print(f'  falls: {"NONE" if len(lowh)==0 else len(lowh)}  h_min={h.min():.3f}  survival(h>0.5)={(h>0.5).mean()*100:.0f}%')
print(f'  stop |vx| last10s = {np.abs(vx[-500:]).mean():.3f}')
dtau = np.abs(np.diff(tau, axis=0))/dt
print(f'  torque rate mean={dtau.mean():.0f} p95={np.percentile(dtau,95):.0f} Nm/s (target<=180)')

print('== 4. TRACKING ==')
for i, (c, s) in enumerate(zip([0.2,0.4,0.6,0.4,0.2,0.0], [i*500 for i in range(6)])):
    e = min(s+500, len(vx))
    print(f'  seg{i} cmd={c:.1f}: vx={vx[s:e].mean():.3f} ({vx[s:e].mean()/c*100 if c else 0:.0f}%) vy={vy[s:e].mean():+.3f}')
print(f'  total travel: dx={px[-1]-px[0]:.2f}m dy={py[-1]-py[0]:.2f}m heading={np.degrees(np.arctan2(py[-1]-py[0],px[-1]-px[0])):.1f}deg')
