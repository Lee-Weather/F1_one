"""Analyze isaac_diag.csv to compute gait metrics for X1 humanoid."""
import sys
import numpy as np
import pandas as pd

def analyze_gait(csv_path):
    df = pd.read_csv(csv_path)
    dt = 0.01  # 100Hz play simulation
    N = len(df)

    # Steady state: skip first 500 steps (5s) for transient
    ss = df.iloc[500:].copy()

    # === 1. Body height ===
    body_height = ss["base_height"].mean()

    # === 2. Average velocity ===
    avg_vel = ss["base_vel_x"].mean()

    # === 3. Foot contact detection (foot_z pattern) ===
    # Foot z threshold for ground contact
    foot_z_l = df["foot_z_l"].values
    foot_z_r = df["foot_z_r"].values

    # Contact force threshold
    force_l = df["foot_force_l"].values
    force_r = df["foot_force_r"].values
    contact_threshold = 5.0

    contact_l = force_l > contact_threshold
    contact_r = force_r > contact_threshold

    # Detect swing onsets (transition from contact to swing)
    def detect_onsets(contact):
        onsets = []
        for i in range(1, len(contact)):
            if not contact[i] and contact[i-1]:  # contact -> swing
                onsets.append(i)
        return np.array(onsets)

    onsets_l = detect_onsets(contact_l)
    onsets_r = detect_onsets(contact_r)

    # === 4. Cycle time (time between consecutive onsets of same foot) ===
    def cycle_times(onsets):
        if len(onsets) < 2:
            return np.array([])
        diffs = np.diff(onsets) * dt
        return diffs

    cycles_l = cycle_times(onsets_l)
    cycles_r = cycle_times(onsets_r)

    # Filter reasonable cycles (0.2s - 2.0s)
    def filter_cycles(cycles, lo=0.2, hi=2.0):
        return cycles[(cycles >= lo) & (cycles <= hi)]

    cycles_l_f = filter_cycles(cycles_l)
    cycles_r_f = filter_cycles(cycles_r)

    cycle_l_mean = np.mean(cycles_l_f) if len(cycles_l_f) > 0 else 0
    cycle_r_mean = np.mean(cycles_r_f) if len(cycles_r_f) > 0 else 0

    # === 5. Step frequency ===
    freq_l = 1.0 / cycle_l_mean if cycle_l_mean > 0 else 0
    freq_r = 1.0 / cycle_r_mean if cycle_r_mean > 0 else 0

    # === 6. Phase offset ===
    # Time between left onset and nearest right onset, normalized by cycle
    def compute_phase(onsets_a, onsets_b, cycle_a, cycle_b):
        if len(onsets_a) < 2 or len(onsets_b) < 2:
            return None
        cycle_mean = (cycle_a + cycle_b) / 2
        if cycle_mean < 0.01:
            return None
        phases = []
        for oa in onsets_a:
            # Find nearest onset in b
            diffs = np.abs(onsets_b - oa)
            min_idx = np.argmin(diffs)
            min_diff = onsets_b[min_idx] - oa
            if abs(min_diff) * dt < cycle_mean:  # reasonable
                phase = ((onsets_b[min_idx] - oa) * dt) / cycle_mean
                phases.append(abs(phase))
        return np.median(phases) if phases else None

    phase_lr = compute_phase(onsets_l, onsets_r, cycle_l_mean, cycle_r_mean)

    # === 7. Foot lift (max height during swing) ===
    def compute_lift(foot_z, contact):
        lifts = []
        i = 0
        while i < len(contact):
            if not contact[i] and (i == 0 or contact[i-1]):  # swing start
                start = i
                while i < len(contact) and not contact[i]:
                    i += 1
                swing_z = foot_z[start:i]
                if len(swing_z) > 0:
                    # Lift = max - min during swing
                    lift = np.max(swing_z) - np.min(swing_z)
                    lifts.append(lift)
            else:
                i += 1
        return np.mean(lifts) if lifts else 0

    lift_l = compute_lift(foot_z_l, contact_l)
    lift_r = compute_lift(foot_z_r, contact_r)

    # === 8. Step length (velocity * cycle_time / 2) ===
    # Each step covers velocity * step_time, where step_time = cycle/2 (stance phase)
    # Actually step_length = velocity * cycle_time (distance per full cycle per foot)
    # Better: step_length ≈ avg_vel * cycle_time / 1 (one full cycle = one step per foot)
    step_len_l = avg_vel * cycle_l_mean if cycle_l_mean > 0 else 0
    step_len_r = avg_vel * cycle_r_mean if cycle_r_mean > 0 else 0

    # === 9. Foot orientation (真实脚朝向：feet yaw 相对 base yaw) ===
    # CSV 由 play_gm.py 记录 foot_yaw_l/foot_yaw_r = feet_euler_xyz[:, :, 2] - base_yaw
    # 注意：raw 值可能超出 [-pi,pi]（如 3.1 跳到 -3.1 的 wrap 跳变），必须 wrap 后再取均值
    def wrap_to_pi_ser(s):
        return ((s + np.pi) % (2 * np.pi)) - np.pi

    if "foot_yaw_l" in ss.columns:
        foot_yaw_l = wrap_to_pi_ser(ss["foot_yaw_l"]).mean()
        foot_yaw_r = wrap_to_pi_ser(ss["foot_yaw_r"]).mean()
    else:
        # 旧 CSV 回退：用 hip_yaw 关节角（X1 的 hip_yaw 轴非竖直，仅粗略）
        foot_yaw_l = ss["dof_pos_2"].mean()
        foot_yaw_r = ss["dof_pos_8"].mean()

    # === Summary ===
    metrics = {
        "body_height": (body_height, "~0.61m", abs(body_height - 0.61) < 0.03),
        "avg_velocity": (avg_vel, "~0.5", abs(avg_vel - 0.5) < 0.1),
        "cycle_l": (cycle_l_mean, "0.55~0.85s", 0.55 <= cycle_l_mean <= 0.85),
        "cycle_r": (cycle_r_mean, "0.55~0.85s", 0.55 <= cycle_r_mean <= 0.85),
        "freq_l": (freq_l, "1.2~1.8", 1.2 <= freq_l <= 1.8),
        "freq_r": (freq_r, "1.2~1.8", 1.2 <= freq_r <= 1.8),
        "step_len_l": (step_len_l, ">=0.30m", step_len_l >= 0.30),
        "step_len_r": (step_len_r, ">=0.30m", step_len_r >= 0.30),
        "lift_l": (lift_l, ">=0.03m", lift_l >= 0.03),
        "lift_r": (lift_r, ">=0.03m", lift_r >= 0.03),
        "foot_yaw_l": (foot_yaw_l, "≈0", abs(foot_yaw_l) < 0.15),
        "foot_yaw_r": (foot_yaw_r, "≈0", abs(foot_yaw_r) < 0.15),
        "phase_offset": (phase_lr if phase_lr else 0, "~0.5", abs((phase_lr if phase_lr else 0) - 0.5) < 0.15),
    }

    print("=" * 70)
    print(f"GAIT ANALYSIS: {csv_path}")
    print("=" * 70)
    print(f"Data points: {N}, steady-state range: [{500*dt:.1f}s, {N*dt:.1f}s]")
    print(f"Left onsets: {len(onsets_l)}, Right onsets: {len(onsets_r)}")
    print(f"Left cycles (filtered): {len(cycles_l_f)}, Right cycles (filtered): {len(cycles_r_f)}")
    print()
    print(f"{'Metric':<20} {'Value':>10} {'Target':>12} {'Pass':>6}")
    print("-" * 52)
    all_pass = True
    for name, (val, target, passed) in metrics.items():
        status = "✅" if passed else "❌"
        if not passed:
            all_pass = False
        print(f"{name:<20} {val:>10.4f} {target:>12} {status:>6}")
    print("-" * 52)
    print(f"{'ALL PASS':<20} {'':>10} {'':>12} {'✅' if all_pass else '❌':>6}")
    print()

    return metrics

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "czy/data/exp0_5/isaac_diag.csv"
    analyze_gait(csv_path)
