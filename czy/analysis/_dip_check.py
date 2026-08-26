import csv
rows = list(csv.DictReader(open(r'e:\X1\F1_one\F1_one\czy\data\exp2.5\isaac_diag.csv')))
sel = [r for r in rows if 10.6 < float(r['time_s']) < 12.6]
for r in sel[::10]:
    print('t=%s h=%.2f v=%+.2f cmd=%.2f yaw=%+.2f' % (
        r['time_s'][:5], float(r['base_height']), float(r['base_vel_x']),
        float(r['command_x']), float(r['base_yaw'])))
