# -*- coding: utf-8 -*-
"""新旧 URDF 对比分析：M_ii 能量法 + FK 对称性 + config 兼容性。

M_ii = 2*KE / ω² ：机身固定、其他关节锁定、关节 i 单位角速度时全部下游
link 的动能（平动 m|ω×r|² + 转动 ω·I·ω，I 变换到世界系）。
对标 czy/diff/GENERAL_JOINT_STEP_DYNAMICS_ANALYSIS_WORKFLOW.md §12 辨识表。
"""
import xml.etree.ElementTree as ET
import numpy as np
import itertools

URDF_OLD = "resources/robots/x1/urdf/X1_12DOF.urdf"
URDF_NEW = "resources/robots/x1/urdf/X1_12DOF_physically_mirrored.urdf"

# config x1_dh_stand_config 的名义关节角（dof 顺序）
DEFAULT = {
    "left_hip_pitch_joint": 0.4, "left_hip_roll_joint": 0.05, "left_hip_yaw_joint": -0.31,
    "left_knee_pitch_joint": 0.49, "left_ankle_pitch_joint": -0.21, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.4, "right_hip_roll_joint": -0.05, "right_hip_yaw_joint": 0.31,
    "right_knee_pitch_joint": 0.49, "right_ankle_pitch_joint": -0.21, "right_ankle_roll_joint": 0.0,
}

def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx

def axis_angle_R(axis, q):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)

class Model:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.links = {}
        for L in root.findall("link"):
            ine = L.find("inertial")
            if ine is None:
                self.links[L.get("name")] = None
                continue
            o = ine.find("origin"); m = ine.find("mass"); i = ine.find("inertia")
            self.links[L.get("name")] = dict(
                com=np.array([float(v) for v in o.get("xyz").split()]),
                mass=float(m.get("value")),
                I=np.array([[float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                            [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                            [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))]]))
        self.joints = {}
        self.children = {}
        for J in root.findall("joint"):
            n = J.get("name")
            o = J.find("origin"); a = J.find("axis")
            self.joints[n] = dict(
                type=J.get("type"), parent=J.find("parent").get("link"), child=J.find("child").get("link"),
                xyz=np.array([float(v) for v in o.get("xyz").split()]),
                R=rpy_to_R(*[float(v) for v in o.get("rpy").split()]),
                axis=np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None)
            self.children.setdefault(J.find("parent").get("link"), []).append(n)

    def fk(self, q):
        """返回 {link: (R_world, p_world)}，从 base_link 出发（base 在原点）。"""
        out = {"base_link": (np.eye(3), np.zeros(3))}
        stack = ["base_link"]
        while stack:
            par = stack.pop()
            R0, p0 = out[par]
            for jn in self.children.get(par, []):
                J = self.joints[jn]
                qj = q.get(jn, 0.0) if J["type"] == "revolute" else 0.0
                Rj = J["R"] @ (axis_angle_R(J["axis"], qj) if J["type"] == "revolute" else np.eye(3))
                out[J["child"]] = (R0 @ Rj, p0 + R0 @ J["xyz"])
                stack.append(J["child"])
        return out

    def downstream_links(self, joint_name):
        """关节 child 起的全部 link 名集合。"""
        seen, stack = set(), [self.joints[joint_name]["child"]]
        while stack:
            l = stack.pop()
            if l in seen: continue
            seen.add(l)
            for jn in self.children.get(l, []):
                stack.append(self.joints[jn]["child"])
        return seen

    def mii(self, joint_name, q):
        """能量法 M_ii：单位 ω 绕关节轴（世界系）。"""
        fk = self.fk(q)
        J = self.joints[joint_name]
        R_child, p_child = fk[J["child"]]
        axis_w = R_child @ (J["axis"] / np.linalg.norm(J["axis"]))
        # 关节原点在 child 系的位置是 0（URDF: child frame = joint frame）
        p_axis = p_child
        KE = 0.0
        for l in self.downstream_links(joint_name):
            info = self.links[l]
            if info is None: continue
            R, p = fk[l]
            com_w = p + R @ info["com"]
            v = np.cross(axis_w, com_w - p_axis)
            I_w = R @ info["I"] @ R.T
            KE += 0.5 * info["mass"] * v @ v + 0.5 * axis_w @ I_w @ axis_w
        return 2.0 * KE  # / ω²(=1)

# 辨识表（WORKFLOW §12）
IDENT = {
    "left_hip_pitch_joint": 0.467, "right_hip_pitch_joint": 0.399,
    "left_hip_roll_joint": 0.389, "right_hip_roll_joint": 0.234,   # 右: 窗口敏感,仅供参考
    "left_hip_yaw_joint": 0.0457, "right_hip_yaw_joint": 0.0369,
    "left_knee_pitch_joint": 0.3626, "right_knee_pitch_joint": 0.3595,
}
OLD_URDF_MII = {  # 辨识当时所用 URDF 的 M_ii（表格 URDF J 列）
    "left_hip_pitch_joint": 0.271, "right_hip_pitch_joint": 0.270,
    "left_hip_roll_joint": 0.485, "right_hip_roll_joint": 0.484,
    "left_hip_yaw_joint": 0.0309, "right_hip_yaw_joint": 0.0309,
    "left_knee_pitch_joint": 0.1127, "right_knee_pitch_joint": 0.1131,
}

mo, mn = Model(URDF_OLD), Model(URDF_NEW)

print("=" * 100)
print("### 1. M_ii 对比（名义位形，机身固定，能量法）")
print("=" * 100)
hdr = f"{'joint':<24}{'旧URDF':>9}{'新URDF':>9}{'辨识J':>9}{'旧/辨识':>9}{'新/辨识':>9}"
print(hdr)
for jn in IDENT:
    a, b = mo.mii(jn, DEFAULT), mn.mii(jn, DEFAULT)
    print(f"{jn.replace('_joint',''):<24}{a:>9.4f}{b:>9.4f}{IDENT[jn]:>9.4f}"
          f"{a/IDENT[jn]:>9.2f}x{b/IDENT[jn]:>9.2f}x")

print()
print("=" * 100)
print("### 2. FK 对称性（名义角，世界系脚心位置, x前/y左/z上）")
print("=" * 100)
for name, m in [("旧", mo), ("新", mn)]:
    fk = m.fk(DEFAULT)
    for foot in ["left_ankle_roll_link", "right_ankle_roll_link"]:
        # 脚心近似: ankle_roll 原点 + 脚 mesh CoM (局部)
        com = m.links[foot]["com"]; R, p = fk[foot]
        print(f"  {name} {foot.replace('_ankle_roll_link','')}: ankle=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}) "
              f"CoM=({(p+R@com)[0]:+.4f},{(p+R@com)[1]:+.4f},{(p+R@com)[2]:+.4f})")
    pl = fk["left_ankle_roll_link"][1]; pr = fk["right_ankle_roll_link"][1]
    print(f"  {name} y和(L+R)={pl[1]+pr[1]:+.5f} (0=镜像), z差={pl[2]-pr[2]:+.5f}, x差={pl[0]-pr[0]:+.5f}")
    print()

print("=" * 100)
print("### 3. 摆动轨迹 FK（final_swing_joint_delta_pos 叠加名义角）")
print("=" * 100)
SWING = dict(DEFAULT)
# config: [hip_p 0.30, hip_r 0.05, hip_y -0.11, knee 0.50, ank_p -0.10, 0] (L) / R: [-0.30,-0.05,+0.11,0.50,-0.10,0]
sw_l = [0.30, 0.05, -0.11, 0.50, -0.10, 0.0]
sw_r = [-0.30, -0.05, 0.11, 0.50, -0.10, 0.0]
lk = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
rk = [s.replace("left", "right") for s in lk]
for k, d in zip(lk, sw_l): SWING[k] = DEFAULT[k] + d
for k, d in zip(rk, sw_r): SWING[k] = DEFAULT[k] + d
for name, m in [("旧", mo), ("新", mn)]:
    fk = m.fk(SWING); base = fk["left_ankle_roll_link"][1] - m.fk(DEFAULT)["left_ankle_roll_link"][1]
    fk0 = m.fk(DEFAULT)
    dl = fk["left_ankle_roll_link"][1] - fk0["left_ankle_roll_link"][1]
    dr = fk["right_ankle_roll_link"][1] - fk0["right_ankle_roll_link"][1]
    print(f"  {name}URDF: L 位移=({dl[0]*100:+.2f},{dl[1]*100:+.2f},{dl[2]*100:+.2f})cm  "
          f"R 位移=({dr[0]*100:+.2f},{dr[1]*100:+.2f},{dr[2]*100:+.2f})cm  "
          f"y和={(dl[1]+dr[1])*100:+.2f}cm z差={(dl[2]-dr[2])*100:+.2f}cm")

print()
print("=" * 100)
print("### 4. config 兼容性: 新 URDF 右腿限位 vs config 默认角")
print("=" * 100)
root = ET.parse(URDF_NEW).getroot()
for J in root.findall("joint"):
    n = J.get("name")
    if n in DEFAULT and J.get("type") == "revolute":
        lim = J.find("limit")
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        q = DEFAULT[n]
        flag = "OK" if lo <= q <= hi else "**超限!**"
        if "right" in n or flag != "OK":
            print(f"  {n:<28} limit=[{lo},{hi}] config默认={q:+.2f} {flag}")
