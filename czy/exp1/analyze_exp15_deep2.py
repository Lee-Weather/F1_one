# exp1.5 part 2: slip & ankle saturation + yaw drift rate per seg
import csv
import numpy as np

ROWS = list(csv.DictReader(open(r'e:\X1\F1_one\F1_one\czy\data\exp1.5\isaac_diag.csv', encoding='utf-8')))
f = lambda k: np.array([float(r[k]) for r in ROWS])
vx, vy, vyaw, yaw = f('base_vel_x'), f('base_vel_y'), f('base_vel_yaw'), f('base_yaw')
fl, fr = f('foot_force_l'), f('foot_force_r')
fzl, fzr = f('foot_z_l'), f('foot_z_r')
dof_pos = np.array([[float(r[f'dof_pos_{j}']) for j in range(12)] for r in ROWS])
dof_tau = np.array([[float(r[f'dof_torque_{j}']) for j in range(12)] for r in ROWS])
SEGS = [(i*500, c) for i, c in enumerate([0.2,0.4,0.6,0.4,0.2,0.0])]
names = ['Lhip_p','Lhip_r','Lhip_y','Lknee','Lankle_p','Lankle_r','Rhip_p','Rhip_r','Rhip_y','Rknee','Rankle_p','Rankle_r']

print('== E. FOOT SLIP PROXY: stance foot vertical micro-bounces ==')
for i, (s, c) in enumerate(SEGS[:5]):
    e = s+500
    # stance foot z variance (slip shows as z chatter while 'in contact')
    lz = fzl[s:e][fl[s:e] > 20].std() if (fl[s:e] > 20).any() else float('nan')
    rz = fzr[s:e][fr[s:e] > 20].std() if (fr[s:e] > 20).any() else float('nan')
    print(f'  seg{i} cmd={c:.1f}: stance z std L={lz*1000:.2f}mm R={rz*1000:.2f}mm')

print()
print('== F. ANKLES ==')
for j in [4,5,10,11]:
    print(f'  {names[j]}: pos {dof_pos[:,j].min():+.2f}..{dof_pos[:,j].max():+.2f}  |tau| mean={np.abs(dof_tau[:,j]).mean():.1f} max={np.abs(dof_tau[:,j]).max():.0f}')

print()
print('== G. YAW DRIFT RATE (deg per second per seg) ==')
for i, (s, c) in enumerate(SEGS):
    e = min(s+500, len(yaw))
    print(f'  seg{i} cmd={c:.1f}: dyaw/dt = {np.degrees(yaw[e-1]-yaw[s])/((e-s)*0.02):+.2f} deg/s')

print()
print('== H. LATERAL v vs FORWARD (seg3/4 turning mechanics) ==')
for i in [3, 4]:
    s = i*500; e = s+500
    turn = 'CW/right' if vyaw[s:e].mean() < 0 else 'CCW/left'
    print(f'  seg{i}: vx={vx[s:e].mean():.3f} vy={vy[s:e].mean():+.3f} vyaw={vyaw[s:e].mean():+.3f} -> turning {turn}')
