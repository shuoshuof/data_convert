import torch

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

# NOITOM2HU_JOINT_MAPPING = {
#     'Hips': 'pelvis_link',
#     'LeftUpLeg': 'left_hip_pitch_link',
#     'LeftLeg': 'left_knee_link',
#     'LeftFoot': 'left_ankle_link',
#     'RightUpLeg': 'right_hip_pitch_link',
#     'RightLeg': 'right_knee_link',
#     'RightFoot': 'right_ankle_link',
#     # 'Torso': 'torso_link',
#     # # 'Spine': 'torso_link',
#     'Spine2': 'torso_link',
#     'Neck': 'neck_link',
#     'LeftArm': 'left_shoulder_roll_link',
#     'LeftForeArm': 'left_elbow_pitch_link',
#     'LeftHand': 'left_wrist_yaw_link',
#     'RightArm': 'right_shoulder_roll_link',
#     'RightForeArm': 'right_elbow_pitch_link',
#     'RightHand': 'right_wrist_yaw_link',
# }
# TODO:反着为什么对
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
    'Spine2': 'torso_link',
    'Neck': 'neck_link',
    'RightArm': 'left_shoulder_roll_link',
    'RightForeArm': 'left_elbow_pitch_link',
    'RightHand': 'left_wrist_yaw_link',
    'LeftArm': 'right_shoulder_roll_link',
    'LeftForeArm': 'right_elbow_pitch_link',
    'LeftHand': 'right_wrist_yaw_link',
}