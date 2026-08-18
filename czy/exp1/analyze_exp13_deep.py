# Deep analysis of exp1.3 isaac_diag.csv
# Segments (500 steps @ 50Hz = 10s each): cmd 0.2/0.4/0.6/0.4/0.2/0.0
import csv
import numpy as np

ROWS = list(csv.DictReader(open(r'e:\X1\F1_one\F1_one\czy\data\exp1.3\isaac_diag.csv', encoding='utf-8')))
f = lambda k: np.array([float(r[k]) for r in ROWS])
cmd, vx, vy, vyaw = f('command_x'), f('base_vel_x'), f('base_vel_y'), f('base_vel_yaw')
h, px, py, yaw = f('base_height'), f('base_pos_x'), f('base_pos_y'), f('base_yaw')
fzl, fzr, fl, fr = f('foot_z_l'), f('foot_z_r'), f('foot_force_l'), f('foot_force_r')
yl, yr = f('foot_yaw_l'), f('foot_yaw_r')
N = len(ROWS); dt = 0.02
SEGS = [(0.2, 0), (0.4, 500), (0.6, 1000), (0.4, 1500), (0.2, 2000), (0.0, 2500)]
CMDS = [0.2, 0.4, 0.6, 0.4, 0.2, 0.0]

def contact(fs, th=20.0):
    return fs > th

def step_periods(fz, on):
    """rising edges of contact = touchdown; period = interval between touchdowns"""
    td = np.where((on[1:] & ~on[:-1]))[0] + 1
    if len(td) < 3: return td, np.array([])
    return td, np.diff(td) * dt

print('=' * 78)
print('A. PER-SEGMENT GAIT METRICS')
print('=' * 78)
print(f"{'seg':>3} {'cmd':>4} {'vx':>6} {'track%':>6} {'T_L(s)':>6} {'T_R(s)':>6} {'stride(m)':>9} {'clr_L':>6} {'clr_R':>6} {'2sup%':>5} {'fly%':>5}")
for i, (c, s) in enumerate(SEGS):
    e = s + 500
    onL, onR = contact(fl[s:e]), contact(fr[s:e])
    _, pL = step_periods(fzl[s:e], onL)
    _, pR = step_periods(fzr[s:e], onR)
    both = (onL & onR).mean() * 100
    fly = (~onL & ~onR).mean() * 100
    v = vx[s:e].mean()
    # stride = single-leg period * speed (approx distance per cycle)
    T = np.median(pL) if len(pL) else np.nan
    stride = T * v if T == T else np.nan
    print(f"{i:>3} {c:>4.1f} {v:>6.3f} {v/c*100 if c else 0:>5.0f}% {np.median(pL) if len(pL) else float('nan'):>6.3f} {np.median(pR) if len(pR) else float('nan'):>6.3f} {stride:>9.3f} {fzl[s:e].max():>6.3f} {fzr[s:e].max():>6.3f} {both:>5.1f} {fly:>5.1f}")

print()
print('=' * 78)
print('B. TRANSITION DYNAMICS (response after each command step)')
print('=' * 78)
trans = [(500, 0.2, 0.4), (1000, 0.4, 0.6), (1500, 0.6, 0.4), (2000, 0.4, 0.2), (2500, 0.2, 0.0)]
print(f"{'t(s)':>5} {'from':>5} {'to':>5} {'t_90%(s)':>8} {'settle_vx':>9} {'overshoot':>9} {'v@2s':>6} {'v@4s':>6} {'fall?':>5}")
for idx, c0, c1 in trans:
    seg = vx[idx:idx + 500]
    target = c1
    # overshoot relative to target
    if c1 > c0:  # accel
        over = (seg.max() - target) if seg.max() > target else 0.0
    else:  # decel
        over = 0.0
    err = np.abs(seg - target)
    t90 = np.argmax(err <= 0.1 * max(target, 0.1)) * dt if (err <= 0.1 * max(target, 0.1)).any() else float('nan')
    # fall detection: height dip within 4s window after switch
    hw = h[idx:idx + 200]
    fall = 'YES' if hw.min() < 0.35 else ''
    print(f"{idx*dt:>5.0f} {c0:>5.1f} {c1:>5.1f} {t90:>8.2f} {seg.mean():>9.3f} {over:>9.2f} {seg[:100].mean():>6.3f} {seg[100:200].mean():>6.3f} {fall:>5}")

print()
print('=' * 78)
print('C. FALL EVENTS (height < 0.35m or |yaw rate| > 1.0)')
print('=' * 78)
lowh = np.where(h < 0.35)[0]
if len(lowh):
    groups = np.split(lowh, np.where(np.diff(lowh) > 25)[0] + 1)
    for g in groups:
        c0 = int(g[0]); c1 = int(g[-1])
        print(f"  t={c0*dt:5.1f}s..{c1*dt:5.1f}s | cmd={cmd[c0]:.1f} | h_min={h[c0:c1].min():.3f} | "
              f"vyaw_max={np.abs(vyaw[c0:c1]).max():.2f} | pos reset {px[c0]:.2f}->{px[c1]:.2f}")
else:
    print('  none')

print()
print('=' * 78)
print('D. STABILITY: vy bias / yaw drift / lateral excursion')
print('=' * 78)
for i, (c, s) in enumerate(SEGS):
    e = s + 500
    lat = py[s:e] - py[s]
    print(f"  seg{i} cmd={c:.1f}: vy_mean={vy[s:e].mean():+.3f} lat_excur={lat.max()-lat.min():.3f}m "
          f"yaw_mean={np.degrees(yaw[s:e]).mean():+.1f} deg")

print()
print('=' * 78)
print('E. JOINT LEVEL (first 6 dof names assumed: hip_pitch L/R, ... placeholder stats)')
print('=' * 78)
dof_pos = np.array([[float(r[f'dof_pos_{j}']) for j in range(12)] for r in ROWS])
dof_vel = np.array([[float(r[f'dof_vel_{j}']) for j in range(12)] for r in ROWS])
dof_tau = np.array([[float(r[f'dof_torque_{j}']) for j in range(12)] for r in ROWS])
print(f"  dof_pos  range: {dof_pos.min():+.2f}..{dof_pos.max():+.2f} rad")
print(f"  dof_vel  p95:   {np.percentile(np.abs(dof_vel), 95):.2f} rad/s  max {np.abs(dof_vel).max():.2f}")
print(f"  torque   p95:   {np.percentile(np.abs(dof_tau), 95):.1f} Nm     max {np.abs(dof_tau).max():.1f}")
# per-segment torque duty
for i, (c, s) in enumerate(SEGS):
    e = s + 500
    print(f"  seg{i} cmd={c:.1f}: tau_mean={np.abs(dof_tau[s:e]).mean():5.1f} Nm  tau_max={np.abs(dof_tau[s:e]).max():6.1f} Nm")

print()
print('=' * 78)
print('F. FOOT FORCES (impact peaks, N)')
print('=' * 78)
for i, (c, s) in enumerate(SEGS):
    e = s + 500
    print(f"  seg{i} cmd={c:.1f}: L_peak={fl[s:e].max():6.0f}  R_peak={fr[s:e].max():6.0f}  "
          f"L_mean(contact)={fl[s:e][contact(fl[s:e])].mean():6.0f}  R_mean(contact)={fr[s:e][contact(fr[s:e])].mean():6.0f}")

print()
print('=' * 78)
print('G. CADENCE-VELOCITY FIT (gait cycle vs speed, exp1 legacy check)')
print('=' * 78)
Ts, vs_ = [], []
for i, (c, s) in enumerate(SEGS[:-1]):
    e = s + 500
    onL = contact(fl[s:e])
    _, pL = step_periods(fzl[s:e], onL)
    if len(pL):
        Ts.append(np.median(pL)); vs_.append(vx[s:e].mean())
for T, v in zip(Ts, vs_):
    print(f"  v={v:.2f} m/s -> cycle T={T:.3f}s (freq {1/T:.1f}Hz, stride {T*v:.3f}m)")
