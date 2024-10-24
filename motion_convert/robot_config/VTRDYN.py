import networkx as nx
VTRDYN_JOINT_NAMES = ['Hips',
                      'RightUpperLeg',
                      'RightLowerLeg',
                      'RightFoot',
                      'RightToe',
                       'LeftUpperLeg',
                       'LeftLowerLeg',
                       'LeftFoot',
                       'LeftToe',
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
VTRDYN_CONNECTIONS = [(0,5),(5,6),(6,7),(7,8),
                      (0,1),(1,2),(2,3),(3,4),
                      (0,9),(9,10),(10,11),(11,12),(12,13),(13,14),
                      (12,19),(19,20),(20,21),(21,22),
                      (12,15),(15,16),(16,17),(17,18)]

vtrdyn_graph = nx.DiGraph()


for i, keypoint_name in enumerate(VTRDYN_CONNECTIONS):
    vtrdyn_graph.add_node(i, label=keypoint_name)

vtrdyn_graph.add_edges_from(VTRDYN_CONNECTIONS)





