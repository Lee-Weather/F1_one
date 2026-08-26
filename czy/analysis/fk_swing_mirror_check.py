"""exp2.4: FK check that the mirrored swing delta (R hip_roll -0.05) gives
laterally mirrored foot motion on the 45-deg tilted hip chain."""
import xml.etree.ElementTree as ET
import numpy as np

URDF = r"e:\X1\F1_one\F1_one\resources\robots\x1\urdf\X1_12DOF.urdf"

DEFAULTS = {
    'left_hip_pitch_joint': 0.4, 'left_hip_roll_joint': 0.05, 'left_hip_yaw_joint': -0.31,
    'left_knee_pitch_joint': 0.49, 'left_ankle_pitch_joint': -0.21, 'left_ankle_roll_joint': 0.0,
    'right_hip_pitch_joint': -0.4, 'right_hip_roll_joint': -0.05, 'right_hip_yaw_joint': 0.31,
    'right_knee_pitch_joint': 0.49, 'right_ankle_pitch_joint': -0.21, 'right_ankle_roll_joint': 0.0,
}
L_JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint']
R_JOINTS = ['right_' + n.split('_', 1)[1] for n in L_JOINTS]


def rpy_to_R(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_R(axis, angle):
    a = np.array(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


tree = ET.parse(URDF)
child_map = {}
for j in tree.getroot().iter('joint'):
    origin = j.find('origin')
    xyz = [float(v) for v in (origin.get('xyz') if origin is not None and origin.get('xyz') else '0 0 0').split()]
    rpy = [float(v) for v in (origin.get('rpy') if origin is not None and origin.get('rpy') else '0 0 0').split()]
    ax = j.find('axis')
    child_map[j.find('child').get('link')] = dict(
        name=j.get('name'),
        xyz=np.array(xyz), R=rpy_to_R(rpy),
        axis=[float(v) for v in (ax.get('xyz') if ax is not None else '0 0 1').split()],
        parent=j.find('parent').get('link'), type=j.get('type'))


def foot_pos(angles, side):
    chain = []
    link = f'{side}_ankle_roll_link'
    while link in child_map:
        jt = child_map[link]
        chain.append(jt)
        link = jt['parent']
    chain.reverse()
    p = np.zeros(3)
    R = np.eye(3)
    for jt in chain:
        p = p + R @ jt['xyz']   # xyz in PARENT frame
        R = R @ jt['R']
        if jt['type'] == 'revolute':
            R = R @ axis_R(jt['axis'], angles.get(jt['name'], 0.0))
    return p


# sanity: default pose mirrored
pl0, pr0 = foot_pos(DEFAULTS, 'left'), foot_pos(DEFAULTS, 'right')
print(f"default pose  L={np.round(pl0,4)}  R={np.round(pr0,4)}  mirrored_ok="
      f"{np.isclose(pl0[1], -pr0[1]) and np.isclose(pl0[2], pr0[2])}")

SWING_L = [0.30, 0.05, -0.11, 0.50, -0.10, 0.0]
for tag, SWING_R in (("exp2.3 same-sign roll  ", [-0.30, 0.05, 0.11, 0.50, -0.10, 0.0]),
                     ("exp2.4 mirrored roll   ", [-0.30, -0.05, 0.11, 0.50, -0.10, 0.0])):
    al, ar = dict(DEFAULTS), dict(DEFAULTS)
    for jn, d in zip(L_JOINTS, SWING_L):
        al[jn] += d
    for jn, d in zip(R_JOINTS, SWING_R):
        ar[jn] += d
    dl = (foot_pos(al, 'left') - pl0) * 100
    dr = (foot_pos(ar, 'right') - pr0) * 100
    print(f"{tag}: L d=({dl[0]:+5.1f},{dl[1]:+5.1f},{dl[2]:+5.1f})  R d=({dr[0]:+5.1f},{dr[1]:+5.1f},{dr[2]:+5.1f}) cm"
          f"  |dy_L - (-dy_R)|={abs(dl[1]+dr[1]):.2f}  lift {dl[2]:.1f}/{dr[2]:.1f}cm")
