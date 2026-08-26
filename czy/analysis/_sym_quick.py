import csv, statistics as st
rows = list(csv.DictReader(open(r'e:\X1\F1_one\F1_one\czy\data\exp2.4\isaac_diag.csv')))
N = len(rows)
pl = [float(r['foot_force_l']) for r in rows]
pr = [float(r['foot_force_r']) for r in rows]
fl = [float(r['foot_z_l']) for r in rows]
fr = [float(r['foot_z_r']) for r in rows]
h = [float(r['base_height']) for r in rows]
DT = 0.01
valid = [h[i] > 0.45 for i in range(N)]
impL = sum(pl[i] for i in range(N) if valid[i]) * DT
impR = sum(pr[i] for i in range(N) if valid[i]) * DT
print('impulse R/L = %.2f (%.0f/%.0f N.s)' % (impR / impL, impL, impR))
# ankle amplitude (robust p95-p05) over valid frames
def amp(idx):
    v = sorted(float(r['dof_pos_%d' % idx]) for i, r in enumerate(rows) if valid[i])
    return v[int(0.95 * len(v))] - v[int(0.05 * len(v))]
aL, aR = amp(4), amp(10)
print('ankle_pitch amp L=%.3f R=%.3f ratio=%.2f' % (aL, aR, aR / aL))
kL, kR = amp(3), amp(9)
print('knee amp L=%.3f R=%.3f ratio=%.2f' % (kL, kR, kR / kL))
# stance duty per foot
dL = sum(1 for i in range(N) if valid[i] and pl[i] > 1.0) / sum(1 for i in range(N) if valid[i])
dR = sum(1 for i in range(N) if valid[i] and pr[i] > 1.0) / sum(1 for i in range(N) if valid[i])
print('stance duty L=%.2f R=%.2f' % (dL, dR))
