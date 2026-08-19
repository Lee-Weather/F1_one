# exp1.5 gait-quality acceptance: swing lift + jitter + red lines (vs exp1.4 baseline)
import csv
import numpy as np

def load(p):
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    f = lambda k: np.array([float(r[k]) for r in rows])
    d = {k: f(k) for k in ['command_x','base_vel_x','base_vel_y','base_height','base_pos_x','base_pos_y',
                            'foot_z_l','foot_z_r','foot_force_l','foot_force_r']}
    d['dof_vel'] = np.array([[float(r[f'dof_vel_{j}']) for j in range(12)] for r in rows])
    d['dof_tau'] = np.array([[float(r[f'dof_torque_{j}']) for j in range(12)] for r in rows])
    return d

def hp(x, win=25):
    k = np.ones(win)/win
    return x - np.convolve(x, k, mode='same')

d = load(r'e:\X1\F1_one\F1_one\czy\data\exp1.5\isaac_diag.csv')
dt = 0.02

print('== EXP1.5 GAIT QUALITY ==')
print('-- A. SWING FOOT LIFT (target >=8cm) --')
for side, fz, on in [('L', d['foot_z_l'], d['foot_force_l']>20), ('R', d['foot_z_r'], d['foot_force_r']>20)]:
    swing = ~on
    edges = np.where(np.diff(swing.astype(int)) == 1)[0] + 1
    peaks = np.array([fz[edges[i]:edges[i+1]].max() for i in range(len(edges)-1) if edges[i+1]-edges[i] > 3])
    print(f'  {side}: peak lift mean={peaks.mean()*100:.1f}cm p10={np.percentile(peaks,10)*100:.1f}cm max={peaks.max()*100:.1f}cm (n={len(peaks)})')

print('-- B. JITTER (target <=180 Nm/s) --')
dv, tau = d['dof_vel'], d['dof_tau']
jerk = np.abs(np.diff(dv, axis=0))/dt
dtau = np.abs(np.diff(tau, axis=0))/dt
print(f'  joint jerk mean={jerk.mean():.1f} p95={np.percentile(jerk,95):.1f} rad/s2')
print(f'  torque rate mean={dtau.mean():.0f} p95={np.percentile(dtau,95):.0f} Nm/s')

print('-- C. CADENCE (target <=110 swings/60s) --')
for side, on in [('L', d['foot_force_l']>20), ('R', d['foot_force_r']>20)]:
    swings = int(np.sum(np.diff((~on).astype(int)) == 1))
    print(f'  {side}: {swings} swings/60s')

print('-- D. RED LINES --')
h = d['base_height']; px, py = d['base_pos_x'], d['base_pos_y']
print(f'  falls (h<0.35): {"NONE" if (h<0.35).sum()==0 else (h<0.35).sum()}  h_min={h.min():.3f}')
print(f'  survival (h>0.5): {(h>0.5).mean()*100:.1f}%')
print(f'  heading: {np.degrees(np.arctan2(py[-1]-py[0], px[-1]-px[0])):.1f} deg over {px[-1]-px[0]:.2f}m')
vx = d['base_vel_x']
print(f'  stop accuracy: |vx| last 10s = {np.abs(vx[-500:]).mean():.3f}')
