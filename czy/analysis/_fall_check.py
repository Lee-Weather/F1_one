import csv
rows = list(csv.DictReader(open(r'e:\X1\F1_one\F1_one\czy\data\exp2.4\isaac_diag.csv')))
for t0 in [11.0, 12.9]:
    sel = [r for r in rows if t0 < float(r['time_s']) < t0 + 1.3]
    print('--- window t=%.1f cmd=%.2f' % (t0, float(sel[0]['command_x'])))
    for r in sel[::10]:
        print('  t=%s h=%.2f v=%+.2f yaw=%+.2f y=%+.2f x=%.2f' % (
            r['time_s'][:5], float(r['base_height']), float(r['base_vel_x']),
            float(r['base_yaw']), float(r['base_pos_y']), float(r['base_pos_x'])))
