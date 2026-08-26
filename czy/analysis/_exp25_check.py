import csv, itertools
rows = list(csv.DictReader(open(r'e:\X1\F1_one\F1_one\czy\data\exp2.5\isaac_diag.csv')))
N = len(rows)
h = [float(r['base_height']) for r in rows]
t = [float(r['time_s']) for r in rows]
yaw = [float(r['base_yaw']) for r in rows]
x = [float(r['base_pos_x']) for r in rows]
y = [float(r['base_pos_y']) for r in rows]
v = [float(r['base_vel_x']) for r in rows]
low = [(t[i], round(h[i], 2)) for i in range(N) if h[i] < 0.45]
print('low_h frames:', low)
print('x: %.2f -> %.2f (net %.2f)' % (x[0], x[-1], x[-1]))
print('y |max| = %.2f' % max(abs(q) for q in y))
print('peak vel = %.2f m/s' % max(v))
n = N // 6
for s in range(6):
    seg = yaw[s*n:(s+1)*n]
    print('seg%d yaw %+.3f -> %+.3f (drift %+.3f)' % (s, seg[0], seg[-1], seg[-1] - seg[0]))
# impulse symmetry
DT = 0.01
pl = [float(r['foot_force_l']) for r in rows]
pr = [float(r['foot_force_r']) for r in rows]
valid = [h[i] > 0.45 for i in range(N)]
impL = sum(pl[i] for i in range(N) if valid[i]) * DT
impR = sum(pr[i] for i in range(N) if valid[i]) * DT
print('impulse R/L = %.2f' % (impR / impL))
