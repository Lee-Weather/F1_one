# Swing height & jitter analysis: exp1.4 vs exp1.3 (user reports: feet lift too low, severe jitter)
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
    """detrend by moving average (0.5s at 50Hz) -> high-freq component"""
    k = np.ones(win)/win
    base = np.convolve(x, k, mode='same')
    return x - base

def analyze(name, d):
    print('='*72); print(name); print('='*72)
    fz_l, fz_r = d['foot_z_l'], d['foot_z_r']
    on_l, on_r = d['foot_force_l'] > 20.0, d['foot_force_r'] > 20.0
    dt = 0.02

    print('-- A. SWING FOOT HEIGHT (lift) --')
    for side, fz, on in [('L', fz_l, on_l), ('R', fz_r, on_r)]:
        swing = ~on
        # per-swing peak height
        edges = np.where(np.diff(swing.astype(int)) == 1)[0] + 1
        peaks = []
        for i in range(len(edges)-1):
            seg = fz[edges[i]:edges[i+1]]
            if len(seg) > 3: peaks.append(seg.max())
        peaks = np.array(peaks)
        if len(peaks):
            print(f'  {side}: swing-peak height mean={peaks.mean()*100:.1f}cm  p10={np.percentile(peaks,10)*100:.1f}cm  max={peaks.max()*100:.1f}cm  (n={len(peaks)} swings)')

    print('-- B. JITTER METRICS --')
    dv = d['dof_vel']; tau = d['dof_tau']
    jerk = np.abs(np.diff(dv, axis=0)) / dt           # rad/s^2
    dtau = np.abs(np.diff(tau, axis=0)) / dt          # Nm/s
    print(f'  joint jerk |dv/dt|: mean={jerk.mean():.1f} p95={np.percentile(jerk,95):.1f} rad/s2')
    print(f'  torque rate |dtau/dt|: mean={dtau.mean():.0f} p95={np.percentile(dtau,95):.0f} Nm/s')
    for k, lbl in [('base_height','base height'), ('base_vel_x','vx'), ('foot_z_l','foot_z L')]:
        h = hp(d[k])
        print(f'  {lbl}: high-freq(>1Hz) std = {h.std():.4f}')
    # dof_vel high-frequency energy ratio per joint (top 4 jittery joints)
    hf_ratio = []
    for j in range(12):
        e_total = np.var(dv[:, j])
        e_hf = np.var(hp(dv[:, j]))
        hf_ratio.append(e_hf/e_total if e_total > 0 else 0)
    top = np.argsort(hf_ratio)[::-1][:4]
    print('  top jittery joints (hf energy ratio):', ', '.join(f'dof{j}={hf_ratio[j]:.2f}' for j in top))

    print('-- C. CONTACT CHATTER --')
    for side, on in [('L', on_l), ('R', on_r)]:
        toggles = np.sum(np.diff(on.astype(int)) != 0)
        print(f'  {side}: contact toggles {toggles} in 60s ({toggles/60:.1f}/s)')

    print('-- D. SPECTRUM (base height & foot_z L, 0-10Hz) --')
    for k in ['base_height', 'foot_z_l']:
        x = d[k] - d[k].mean()
        ps = np.abs(np.fft.rfft(x))**2
        fr = np.fft.rfftfreq(len(x), dt)
        m = (fr > 0.5) & (fr < 10)
        if m.any():
            peak = fr[m][np.argmax(ps[m])]
            print(f'  {k}: dominant osc {peak:.2f} Hz')

for name, path in [('EXP1.4', r'e:\X1\F1_one\F1_one\czy\data\exp1.4\isaac_diag.csv'),
                   ('EXP1.3', r'e:\X1\F1_one\F1_one\czy\data\exp1.3\isaac_diag.csv')]:
    analyze(name, load(path))
