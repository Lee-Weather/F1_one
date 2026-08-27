# -*- coding: utf-8 -*-
"""exp2.8 FK 冒烟：新 URDF + 新 config 角度 vs 旧 URDF + 旧角度，站姿+摆动轨迹位形必须一致。"""
import sys, os
sys.path.insert(0, os.path.join("czy", "analysis"))
import numpy as np
from urdf_mii_check import Model  # noqa: E402 (import 会打印主报告, 忽略)

mo = Model("resources/robots/x1/urdf/X1_12DOF.urdf")
mn = Model("resources/robots/x1/urdf/X1_12DOF_physically_mirrored.urdf")

OLD_Q = {"left_hip_pitch_joint": 0.4, "left_hip_roll_joint": 0.05, "left_hip_yaw_joint": -0.31,
         "left_knee_pitch_joint": 0.49, "left_ankle_pitch_joint": -0.21, "left_ankle_roll_joint": 0.0,
         "right_hip_pitch_joint": -0.4, "right_hip_roll_joint": -0.05, "right_hip_yaw_joint": 0.31,
         "right_knee_pitch_joint": 0.49, "right_ankle_pitch_joint": -0.21, "right_ankle_roll_joint": 0.0}
NEW_Q = dict(OLD_Q)
NEW_Q["right_ankle_pitch_joint"] = 0.21  # 唯一翻转: 踝 axis 翻转自洽

fo, fn = mo.fk(OLD_Q), mn.fk(NEW_Q)

def com(fk, m, foot):
    R, p = fk[foot]
    return p + R @ m.links[foot]["com"]

ok = True
print("=== 名义站姿 脚CoM 一致性 ===")
for foot in ["left_ankle_roll_link", "right_ankle_roll_link"]:
    co, cn = com(fo, mo, foot), com(fn, mn, foot)
    d = np.linalg.norm(co - cn) * 100
    ok &= d < 0.5
    print(f"{foot.replace('_ankle_roll_link',''):<6} d={d:.2f}cm  旧=({co[0]*100:+.2f},{co[1]*100:+.2f},{co[2]*100:+.2f}) 新=({cn[0]*100:+.2f},{cn[1]*100:+.2f},{cn[2]*100:+.2f})")

SW_O, SW_N = dict(OLD_Q), dict(NEW_Q)
sw_l = [0.30, 0.05, -0.11, 0.50, -0.10, 0.0]
sw_r = [-0.30, -0.05, 0.11, 0.50, 0.10, 0.0]  # 踝翻转, 膝不翻
lk = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
      "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
rk = [k.replace("left", "right") for k in lk]
for k, dv in zip(lk, sw_l):
    SW_O[k] += dv; SW_N[k] += dv
for k, dv in zip(rk, sw_r):
    SW_O[k] += dv; SW_N[k] += dv
fwo, fwn = mo.fk(SW_O), mn.fk(SW_N)
print("=== 摆动峰值脚位移一致性 ===")
for foot in ["left_ankle_roll_link", "right_ankle_roll_link"]:
    do = fwo[foot][1] - fo[foot][1]
    dn = fwn[foot][1] - fn[foot][1]
    d = np.linalg.norm(do - dn) * 100
    ok &= d < 0.5
    print(f"{foot.replace('_ankle_roll_link','')}摆动 d={d:.2f}cm  旧=({do[0]*100:+.2f},{do[1]*100:+.2f},{do[2]*100:+.2f}) 新=({dn[0]*100:+.2f},{dn[1]*100:+.2f},{dn[2]*100:+.2f})")

print("SMOKE_PASS" if ok else "SMOKE_FAIL")
