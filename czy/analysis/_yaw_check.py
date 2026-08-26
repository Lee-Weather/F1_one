import csv, itertools
rows = list(csv.DictReader(open(r'e:\X1\F1_one\F1_one\czy\data\exp2.4\isaac_diag.csv')))
h = [float(r['base_height']) for r in rows]
t = [float(r['time_s']) for r in rows]
yaw = [float(r['base_yaw']) for r in rows]
x = [float(r['base_pos_x']) for r in rows]
y = [float(r['base_pos_y']) for r in rows]
low = [(t[i], round(h[i], 2)) for i in range(len(h)) if h[i] < 0.45]
runs = []
for k, g in itertools.groupby(enumerate(low), lambda kv: kv[0] - kv[1][0]):
    g = list(g)
    runs.append((round(g[0][1][0], 2), round(g[-1][1][0], 2)))
print('low_h runs:', runs)
print('x range: %.2f -> %.2f (net %.2f m)' % (x[0], x[-1], x[-1] - x[0]))
print('y range: %.2f -> %.2f (|max| %.2f)' % (y[0], y[-1], max(abs(v) for v in y)))
print('yaw: start %.3f end %.3f min %.3f max %.3f' % (yaw[0], yaw[-1], min(yaw), max(yaw)))
# per-segment yaw drift (6 segs x 5s)
n = len(yaw) // 6
for s in range(6):
    seg = yaw[s*n:(s+1)*n]
    print('seg%d: yaw %+.3f -> %+.3f (drift %+.3f rad)' % (s, seg[0], seg[-1], seg[-1]-seg[0]))
