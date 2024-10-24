import networkx as nx
NOITOM_BODY_NAMES = ["Head",
                      "Truncus",
                      "Hip",
                      "LeftCollar",
                      "LeftUpArm",
                      "LeftLowArm",
                      "LeftHand",
                      "RightCollar",
                      "RightUpArm",
                      "RightLowArm",
                      "RightHand",
                      "LeftUpLeg",
                      "LeftLowLeg",
                      "LeftFoot",
                      "RightUpLeg",
                      "RightLowLeg",
                      "RightFoot"]

NOITOM_JOINT_NAMES = ['Hips',
                      'RightUpLeg',
                      'RightLeg',
                      'RightFoot',
                      'LeftUpLeg',
                      'LeftLeg',
                      'LeftFoot',
                      'Spine',
                      'Spine1',
                      'Spine2',
                      'Neck',
                      'Neck1',
                      'Head',
                      'RightShoulder',
                      'RightArm',
                      'RightForeArm',
                      'RightHand',
                      'LeftShoulder',
                      'LeftArm',
                      'LeftForeArm',
                      'LeftHand']

NOITOM_CONNECTIONS = [(0,1),(1,2),(2,3),
                      (0,4),(4,5),(5,6),
                      (0,7),(7,8),(8,9),(9,10),(10,11),(11,12),
                      (8,13),(13,14),(14,15),(15,16),
                      (8,17),(17,18),(18,19),(19,20)]



noitom_graph = nx.DiGraph()

for i, keypoint_name in enumerate(NOITOM_JOINT_NAMES):
    noitom_graph.add_node(i, label=keypoint_name)

noitom_graph.add_edges_from(NOITOM_CONNECTIONS)

parent_indices = [-1]+[connection[0] for connection in NOITOM_CONNECTIONS]

if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('test_data/noitom_mocap_data/take002.csv')
    print({name: idx for idx,name in enumerate(NOITOM_JOINT_NAMES)})
    for key in data.columns:
        name = key.split('-')[0]
        if name not in NOITOM_JOINT_NAMES:
            NOITOM_JOINT_NAMES.append(name)

