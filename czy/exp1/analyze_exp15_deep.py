# Deep analysis of exp1.5 isaac_diag.csv: why lift stuck at 6.1cm, why seg4 drifts -26deg
import csv
import numpy as np

ROWS = list(csv.DictReader(open(r'e:\X1\F1_one\F1_one\czy\data\exp1.5\isaac_diag.csv', encoding='utf-8')))
f = lambda k: np.array([float(r[k]) for r in ROWS])
cmd, vx, vy, vyaw = f('command_x'), f('base_vel_x'), f('base_vel_y'), f('base_vel_yaw')
h, px, py, yaw = f('base_height'), f('base_pos_x'), f('base_pos_y'), f('base_yaw')
fzl, fzr, fl, fr = f('foot_z_l'), f('foot_z_r'), f('foot_force_l'), f('foot_force_r')
yl, yr = f('foot_yaw_l'), f('foot_yaw_r')
dof_pos = np.array([[float(r[f'dof_pos_{j}']) for j in range(12)] for r in ROWS])
dof_tau = np.array([[float(r[f'dof_torque_{j}']) for j in range(12)] for r in ROWS])
N = len(ROWS); dt = 0.02
SEGS = [(i*500, c) for i, c in enumerate([0.2,0.4,0.6,0.4,0.2,0.0])]
onL, onR = fl > 20, fr > 20

print('== A. LIFT PROFILE: where does 6.1cm come from? ==')
# per-segment lift stats
for i, (s, c) in enumerate(SEGS):
    e = s+500
    for side, fz, on in [('L', fzl, onL), ('R', fzr, onR)]:
        swing = ~on[s:e]
        edges = np.where(np.diff(swing.astype(int))==1)[0]+1
        peaks = [fz[s+edges[k]:s+edges[k+1]].max() for k in range(len(edges)-1) if edges[k+1]-edges[k]>3]
        if peaks:
            print(f'  seg{i} cmd={c:.1f} {side}: mean={np.mean(peaks)*100:.1f}cm max={max(peaks)*100:.1f}cm n={len(peaks)}')

print()
print('== B. SWING TIMING: lift vs cycle phase (is there time to lift higher?) ==')
# per-segment swing duration
for i, (s, c) in enumerate(SEGS[:5]):
    e = s+500
    durs = []
    for on in [onL, onR]:
        sw = ~on[s:e]
        edges = np.where(np.diff(sw.astype(int))==1)[0]+1
        durs += list(np.diff(edges)) if len(edges)>1 else []
    if durs:
        print(f'  seg{i} cmd={c:.1f}: swing dur mean={np.mean(durs)*dt:.3f}s (n={len(durs)}) -> lift speed ~{0.061/np.mean(durs):.2f} m/s avg')

print()
print('== C. SEG4 DRILLDOWN (-26 deg drift) ==')
s, e = 2000, 2500
print(f'  yaw: start={np.degrees(yaw[s]):.1f} end={np.degrees(yaw[e-1]):.1f} min={np.degrees(yaw[s:e]).min():.1f} max={np.degrees(yaw[s:e]).max():.1f}')
print(f'  vyaw mean={vyaw[s:e].mean():.3f} rad/s; vy mean={vy[s:e].mean():.3f}')
print(f'  foot yaw rel: L mean={np.degrees(yl[s:e]).mean():.1f} R mean={np.degrees(yr[s:e]).mean():.1f} (neg = toe-in?)')
# step length asymmetry
pl_x = f('base_pos_x'); 
strides_l, strides_r = [], []
for on, fz in [(onL, fzl), (onR, fzr)]:
    td = np.where(np.diff(on[s:e].astype(int))==1)[0]+1
    for k in range(len(td)-1):
        a, b = s+td[k], s+td[k+1]
        strides = px[b]-px[a]
# torque asymmetry seg4
tl = np.abs(dof_tau[s:e, 0:6]).mean(axis=0).mean()
tr = np.abs(dof_tau[s:e, 6:12]).mean(axis=0).mean()
print(f'  torque |L| mean={tl:.1f} vs |R| mean={tr:.1f} Nm -> asym {(tl-tr)/(tl+tr)*100:+.1f}%')
# contact time asymmetry
cl = onL[s:e].mean()*100; cr = onR[s:e].mean()*100
print(f'  contact share: L={cl:.1f}% R={cr:.1f}%')
# lateral velocity during seg4
print(f'  vy (body frame est): {vy[s:e].mean():+.3f} -> drifts {"right" if vy[s:e].mean()<0 else "left"}')

print()
print('== D. SEGWISE YAW + HEADING (all segs, drift origin) ==')
for i, (s, c) in enumerate(SEGS):
    e = s+500
    ddx, ddy = px[e]-px[s], py[e]-py[s]
    if ddx**2+ddy**2 > 0.01:
        print(f'  seg{i} cmd={c:.1f}: yaw_start={np.degrees(yaw[s]):+.1f} yaw_end={np.degrees(yaw[e-1]):+.1f} heading={np.degrees(np.arctan2(ddy,ddx)):+.1f}')

print()
print('== E. STANCE FOOT SLIP (is seg4 drifting from foot sliding?) ==')
# stance foot world-frame x velocity: check foot markers unavailable; proxy: base vx while double-support
print('  (proxy) seg4 vx std=%.3f, min=%.3f, max=%.3f' % (vx[2000:2500].std(), vx[2000:2500].min(), vx[2000:2500].max()))
print('  seg4 contact forces: L peak=%.0f R peak=%.0f' % (fl[2000:2500].max(), fr[2000:2500].max()))

print()
print('== F. ANKLE JOINT USAGE (are ankles saturated?) ==')
names = ['Lhip_p','Lhip_r','Lhip_y','Lknee','Lankle_p','Lankle_r','Rhip_p','Rhip_r','Rhip_y','Rknee','Rankle_p','Rankle_r']
for j in [4,5,10,11]:
    print(f'  {names[j]}: pos range {dof_pos[:,j].min():+.2f}..{dof_pos[:,j].max():+.2f}  |tau| mean={np.abs(dof_tau[:,j]).mean():.1f} max={np.abs(dof_tau[:,j]).max():.0f}')
