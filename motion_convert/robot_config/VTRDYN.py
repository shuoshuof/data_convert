import networkx as nx
VTRDYN_JOINT_NAMES = ['Hips',
                      'RightUpperLeg',
                      'RightLowerLeg',
                      'RightFoot',
                      # 'RightToe',
                       'LeftUpperLeg',
                       'LeftLowerLeg',
                       'LeftFoot',
                       # 'LeftToe',
                       'Spine',
                       'Spine1',
                       'Spine2',
                       'Spine3',
                       'Neck',
                       'Head',
                       'RightShoulder',
                       'RightUpperArm',
                       'RightLowerArm',
                       'RightHand',
                       'LeftShoulder',
                       'LeftUpperArm',
                       'LeftLowerArm',
                       'LeftHand']

# VTRDYN_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),
#                       (0,5),(5,6),(6,7),(7,8),
#                       (0,9),(9,10),(10,11),(11,12),(12,13),(13,14),
#                       (12,19),(19,20),(20,21),(21,22),
#                       (12,15),(15,16),(16,17),(17,18)]
VTRDYN_CONNECTIONS = [(0,1),(1,2),(2,3),
                      (0,4),(4,5),(5,6),
                      (0,7),(7,8),(8,9),(9,10),(10,11),(11,12),
                      (10,13),(13,14),(14,15),(15,16),
                      (10,17),(17,18),(18,19),(19,20),]

vtrdyn_graph = nx.DiGraph()


for i, keypoint_name in enumerate(VTRDYN_CONNECTIONS):
    vtrdyn_graph.add_node(i, label=keypoint_name)

vtrdyn_graph.add_edges_from(VTRDYN_CONNECTIONS)


vtrdyn_parent_indices = [-1] + [connection[0] for connection in VTRDYN_CONNECTIONS]


VTRDYN_JOINT_NAMES_LITE = ['Hips',
                          'RightUpperLeg',
                          'RightLowerLeg',
                          'RightFoot',
                           'LeftUpperLeg',
                           'LeftLowerLeg',
                           'LeftFoot',
                           'Spine',
                           'Spine1',
                           'Neck',
                           'Head',
                           'RightShoulder',
                           'RightUpperArm',
                           'RightLowerArm',
                           'RightHand',
                           'LeftShoulder',
                           'LeftUpperArm',
                           'LeftLowerArm',
                           'LeftHand']

VTRDYN_CONNECTIONS_LITE = [(0,1),(1,2),(2,3),
                      (0,4),(4,5),(5,6),
                      (0,7),(7,8),(8,9),(9,10),
                      (8,11),(11,12),(12,13),(13,14),
                      (8,15),(15,16),(16,17),(17,18),]

vtrdyn_lite_graph = nx.DiGraph()

for i, keypoint_name in enumerate(VTRDYN_JOINT_NAMES_LITE):
    vtrdyn_lite_graph.add_node(i, label=keypoint_name)

vtrdyn_lite_graph.add_edges_from(VTRDYN_CONNECTIONS_LITE)