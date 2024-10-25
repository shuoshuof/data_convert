import torch
import networkx as nx

Hu_DOF_IDX_MAPPING = [
    2, 0, 1, 1, 1, 0,
    2, 0, 1, 1, 1, 0,
    2,
    1, 0, 2, 1, 0, 1, 2, 1, 1,
    1, 0, 2, 1, 0, 1, 2, 1, 1,
    2, ]
Hu_DOF_LOWER = torch.tensor([
    -0.1745, -0.3491, -1.9635, 0.0997, -0.6981, -0.3665,
    -0.1745, -0.3491, -1.9635, 0.0997, -0.6981, -0.3665,
    -1.0,
    -3.1416, 0., -1.5708, 0., -1.5708, -0.7854, -0.7854, 0., -0.044,
    -3.1416, -1.5708, -1.5708, 0., -1.5708, -0.7854, -0.7854, 0., -0.044,
    -1.0])
Hu_DOF_UPPER = torch.tensor([
    0.1745, 0.3491, 1.9635, 2.618, 0.6981, 0.3665,
    0.1745, 0.3491, 1.9635, 2.618, 0.6981, 0.3665,
    1.0,
    1.0472, 1.5708, 1.5708, 1.5708, 1.5708, 0.7854, 0.7854, 0.044, 0.,
    1.0472, 0., 1.5708, 1.5708, 1.5708, 0.7854, 0.7854, 0.044, 0.,
    1.0])

SMPL2HU_JOINT_MAPPING = {
    'Pelvis': 'pelvis_link',
    'L_Hip': 'left_hip_pitch_link',
    'L_Knee': 'left_knee_link',
    'L_Ankle': 'left_ankle_link',
    'R_Hip': 'right_hip_pitch_link',
    'R_Knee': 'right_knee_link',
    'R_Ankle': 'right_ankle_link',
    # 'Torso': 'torso_link',
    # # 'Spine': 'torso_link',
    'Chest': 'torso_link',
    'Head': 'neck_link',
    'L_Shoulder': 'left_shoulder_roll_link',
    'L_Elbow': 'left_elbow_pitch_link',
    'L_Wrist': 'left_wrist_yaw_link',
    'R_Shoulder': 'right_shoulder_roll_link',
    'R_Elbow': 'right_elbow_pitch_link',
    'R_Wrist': 'right_wrist_yaw_link',
}

NOITOM2HU_JOINT_MAPPING = {
    'Hips': 'pelvis_link',
    'LeftUpLeg': 'left_hip_pitch_link',
    'LeftLeg': 'left_knee_link',
    'LeftFoot': 'left_ankle_link',
    'RightUpLeg': 'right_hip_pitch_link',
    'RightLeg': 'right_knee_link',
    'RightFoot': 'right_ankle_link',
    # 'Torso': 'torso_link',
    # # 'Spine': 'torso_link',
    'Spine1': 'torso_link',
    'Neck': 'neck_link',
    'LeftArm': 'left_shoulder_roll_link',
    'LeftForeArm': 'left_elbow_pitch_link',
    'LeftHand': 'left_wrist_yaw_link',
    'RightArm': 'right_shoulder_roll_link',
    'RightForeArm': 'right_elbow_pitch_link',
    'RightHand': 'right_wrist_yaw_link',
}

# # # TODO:反着为什么对?
# NOITOM2HU_JOINT_MAPPING = {
#     'Hips': 'pelvis_link',
#     # 'RightUpLeg': 'left_hip_pitch_link',
#     # 'RightLeg': 'left_knee_link',
#     # 'RightFoot': 'left_ankle_link',
#     # 'LeftUpLeg': 'right_hip_pitch_link',
#     # 'LeftLeg': 'right_knee_link',
#     # 'LeftFoot': 'right_ankle_link',
#     # 'Torso': 'torso_link',
#     # # 'Spine': 'torso_link',
#     'Spine1': 'torso_link',
#     'Neck': 'neck_link',
#     'RightArm': 'left_shoulder_roll_link',
#     'RightForeArm': 'left_elbow_pitch_link',
#     'RightHand': 'left_wrist_yaw_link',
#     'LeftArm': 'right_shoulder_roll_link',
#     'LeftForeArm': 'right_elbow_pitch_link',
#     'LeftHand': 'right_wrist_yaw_link',
# }



HU_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),
                  (0,7),(7,8),(8,9),(9,10),(10,11),(11,12),
                  (0,13),
                  (13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),
                  (13,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),
                  (13,32)]

HU_JOINT_NAMES = [i for i in range(33)]

hu_graph = nx.DiGraph()

for i, keypoint_name in enumerate(HU_JOINT_NAMES):
    hu_graph.add_node(i, label=keypoint_name)

hu_graph.add_edges_from(HU_CONNECTIONS)