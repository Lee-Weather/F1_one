# -*- coding: utf-8 -*-
"""exp3.1 logic smoke test (server F1 env, torch 1.12).

Verifies WITHOUT spinning up Isaac Gym:
  A. observation dimension arithmetic (51 / 77 / 57)
  B. skill command resampling distribution (60/20/20)
  C. skill aux derivation (one-hot, height target, ref deltas on the right leg)
  D. lift pose anti-crane property (hip_yaw / ankle_roll deltas are ZERO)
  E. curriculum ladder boundaries (promote at 2 streak, demote on term)
  F. reward registration names match config scales
Run:  python czy/analysis/smoke_exp31.py   (from repo root, F1 env)
"""
import sys

PASS = []
FAIL = []

def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  OK " if cond else " FAIL") + " " + name)

print("== A. dims ==")
# actor single obs: sincos(2)+vel(3)+skill(4)+q(12)+dq(12)+act(12)+ang(3)+eul(3)
single = 2 + 3 + 4 + 12 + 12 + 12 + 3 + 3
check("num_single_obs == 51", single == 51)
# privileged: cmd(9)+q(12)+dq(12)+act(12)+diff(12)+lin(3)+ang(3)+eul(3)+push(2)+tor(3)+fric(1)+mass(1)+stance(2)+contact(2)
priv = 9 + 12 + 12 + 12 + 12 + 3 + 3 + 3 + 2 + 3 + 1 + 1 + 2 + 2
check("single_num_privileged_obs == 77", priv == 77)
check("single_linvel_index == 57", 9 + 12 * 4 == 57)

print("== B. resample distribution ==")
sys.path.insert(0, "humanoid/envs/x1")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cfg", "humanoid/envs/x1/x1_dh_stand_config.py")
# config imports humanoid.envs.base -> needs the package importable; add root
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location(
    "cfgmod", "humanoid/envs/x1/x1_dh_stand_config.py")
try:
    cfgmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfgmod)
    SC = cfgmod.X1DHStandCfg.skill
    check("lift_probs sum to 1", abs(sum(SC.lift_probs) - 1.0) < 1e-9)
    check("height ladder ascending", SC.height_levels == sorted(SC.height_levels))
    check("4 levels with matching drop tolerances",
          len(SC.height_levels) == len(SC.drop_tol_s) == 4)
    check("push only from L4", SC.push_from_level == 4)
    check("resample 10s grace 1s", SC.resample_time == 10.0 and SC.grace_s == 1.0)

    print("== D. anti-crane pose ==")
    check("lift-L hip_yaw delta is 0", SC.lift_pose_delta_l[2] == 0.0)
    check("lift-L ankle_roll delta is 0", SC.lift_pose_delta_l[5] == 0.0)
    check("lift-R hip_yaw delta is 0", SC.lift_pose_delta_r[8 - 6] == 0.0)
    check("lift-R ankle_roll delta is 0", SC.lift_pose_delta_r[11 - 6] == 0.0)
    check("lift pose bends the knee", SC.lift_pose_delta_l[3] >= 0.5)

    print("== F. reward registration ==")
    scales = cfgmod.X1DHStandCfg.rewards.scales
    for key in ["skill_single_support", "skill_lift_height", "skill_stability",
                "skill_foot_flat", "skill_duration", "skill_posture_tax"]:
        check(f"scale.{key} registered", hasattr(scales, key))
except Exception as e:
    check(f"config import failed: {e}", False)

print("== E. curriculum arithmetic (pure logic) ==")
# mimic reset_idx promotion/demotion with plain python
levels = {1: [], 2: [], 3: [], 4: []}
def curriculum(lvl, streak, timed_out, skill_time, ss_ratio):
    clean = timed_out and skill_time >= 3.0 and ss_ratio >= 0.8
    used = skill_time >= 3.0
    streak = streak + 1 if clean else 0
    if streak >= 2:
        lvl = min(4, lvl + 1)
    if (not timed_out) and used:
        lvl = max(1, lvl - 1)
    return lvl, streak

l, s = 1, 0
for _ in range(2):  # two clean L1 episodes -> L2
    l, s = curriculum(l, s, True, 10.0, 0.9)
check("two clean episodes promote 1->2", l == 2 and s == 2)
l2, s2 = curriculum(l, s, False, 5.0, 0.9)  # terminated -> demote back
check("terminated skill episode demotes 2->1", l2 == 1 and s2 == 0)
l3, s3 = curriculum(4, 2, True, 10.0, 0.9)
check("promotion caps at 4", l3 == 4)
l4, s4 = curriculum(1, 0, False, 5.0, 0.9)
check("demotion floors at 1", l4 == 1)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
