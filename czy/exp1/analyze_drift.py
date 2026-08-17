# Check lateral drift direction: world trajectory curvature analysis
import csv
import numpy as np

CSV_PATH = r"e:\X1\F1_one\F1_one\czy\data\exp1\isaac_diag.csv"
DT = 0.02
SEGMENTS = [(0, 500, 0.2), (500, 1000, 0.4), (1000, 1500, 0.6),
            (1500, 2000, 0.4), (2000, 2500, 0.2), (2500, 3000, 0.0)]


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        keys = ["command_x", "base_vel_x", "base_vel_y", "base_vel_yaw",
                "base_pos_x", "base_pos_y", "base_yaw",
                "foot_force_l", "foot_force_r", "foot_yaw_l", "foot_yaw_r", "base_height"]
        data = {k: [] for k in keys}
        for row in r:
            for k in keys:
                data[k].append(float(row[k]))
    return {k: np.array(v) for k, v in data.items()}


def main():
    d = load(CSV_PATH)
    x, y = d["base_pos_x"], d["base_pos_y"]

    print("== full trajectory (world frame, 60s) ==")
    print(f"start=({x[0]:.2f},{y[0]:.2f})  end=({x[-1]:.2f},{y[-1]:.2f})")
    print(f"total displacement: dx={x[-1]-x[0]:.2f} m, dy={y[-1]-y[0]:.2f} m")
    ang = np.degrees(np.arctan2(y[-1] - y[0], x[-1] - x[0]))
    print(f"overall walking direction: {ang:.1f} deg (0=+X world)")
    print()
    # 5s checkpoints
    print("== trajectory checkpoints (every 2s) ==")
    print(f"{'t(s)':<6}{'x':<8}{'y':<8}{'yaw°':<8}{'heading_of_path°':<16}")
    for t in range(0, 60, 2):
        i = min(t * 50, 2999)
        if i == 0:
            hx, hy = 0.0, 0.0
        else:
            # path direction over last 1s
            j = max(0, i - 50)
            hx = np.degrees(np.arctan2(y[i] - y[j], x[i] - x[j]))
        print(f"{t:<6}{x[i]:<8.2f}{y[i]:<8.2f}{np.degrees(d['base_yaw'][i]):<8.1f}{hx:<16.1f}")

    print("\n== per segment: body-yaw at start/end, path direction ==")
    for n, (s, e, cmd) in enumerate(SEGMENTS):
        # skip reset: find jumps
        jumps = np.hypot(np.diff(x[s:e]), np.diff(y[s:e])) > 0.5
        p0 = np.degrees(np.arctan2(y[e-1] - y[s], x[e-1] - x[s])) if not jumps.any() else float("nan")
        yaw0 = np.degrees(d["base_yaw"][s])
        yaw1 = np.degrees(d["base_yaw"][e-1])
        print(f"seg{n} cmd={cmd:.1f}: body_yaw {yaw0:.1f}->{yaw1:.1f} deg | net path dir={p0:.1f} deg"
              f"{' [RESET]' if jumps.any() else ''}")

    # lateral velocity bias in body frame
    print("\n== body-frame lateral velocity bias (vy in body frame) ==")
    for n, (s, e, cmd) in enumerate(SEGMENTS):
        vy = d["base_vel_y"][s:e]
        print(f"seg{n} cmd={cmd:.1f}: mean_vy={vy.mean():+.3f} m/s (positive=left)")

    # foot yaw asymmetry
    print("\n== foot yaw asymmetry (rel to base, deg; + = toe-out left?) ==")
    for n, (s, e, cmd) in enumerate(SEGMENTS):
        yl = np.degrees(d["foot_yaw_l"][s:e])
        yr = np.degrees(d["foot_yaw_r"][s:e])
        print(f"seg{n} cmd={cmd:.1f}: L mean={yl.mean():+.1f} R mean={yr.mean():+.1f}")


if __name__ == "__main__":
    main()
