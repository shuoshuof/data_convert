import networkx as nx


COCO18_KEYPOINTS = [
    "Nose", "NECK", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_HIP",
    "RIGHT_KNEE", "	RIGHT_ANKLE", "LEFT_HIP", "	LEFT_KNEE",
    "LEFT_ANKLE", "RIGHT_EYE", "LEFT_EYE", "RIGHT_EAR", "LEFT_EAR"
]

COCO18_CONNECTIONS = [
    (0, 1), (0, 14), (0, 15), (1, 2), (1, 5), (1, 8), (1, 11), (2, 3),
    (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (11, 12), (12, 13), (14, 16),
    (15, 17)
]

coco18_graph = nx.DiGraph()

for i, keypoint_name in enumerate(COCO18_KEYPOINTS):
    coco18_graph.add_node(i, label=keypoint_name)

coco18_graph.add_edges_from(COCO18_CONNECTIONS)


COCO34_KEYPOINTS = [
    "PELVIS", "NAVAL_SPINE", "CHEST_SPINE", "NECK", "LEFT_CLAVICLE",
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HAND",
    "LEFT_HANDTIP", "	LEFT_THUMB", "RIGHT_CLAVICLE", "	RIGHT_SHOULDER",
    "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_HAND", "RIGHT_HANDTIP", "RIGHT_THUMB",
    "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE","LEFT_FOOT","RIGHT_HIP",
    "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT", "HEAD", "NOSE", "LEFT_EYE",
    "LEFT_EAR", "RIGHT_EYE", "RIGHT_EAR", "LEFT_HEEL", "RIGHT_HEEL",
]

# COCO34_CONNECTIONS = [
#     (0,1),(1,2),
#     (2,3),(3,26),(26,27),(27,28),(28,29),(27,30),(30,31),
#     (2,4),(4,5),(5,6),(6,7),(7,8),(8,9),(7,10),
#     (2,11),(11,12),(12,13),(13,14),(14,15),(15,16),(14,17),
#     (0,18),(18,19),(19,20),(20,21),(20,32),
#     (0,22),(22,23),(23,24),(24,25),(24,33)
# ]
COCO34_CONNECTIONS = [
    (0,1),(1,2),
    # (2,3),(3,26),(26,27),(27,28),(28,29),(27,30),(30,31),
    (2,4),(4,5),(5,6),(6,7),(7,8),(8,9),(7,10),
    (2,11),(11,12),(12,13),(13,14),(14,15),(15,16),(14,17),
    (0,18),(18,19),(19,20),(20,21),(20,32),
    (0,22),(22,23),(23,24),(24,25),(24,33)
]
#不包含头部
COCO34_LENGTH = {
    (0,1):0.2,
    (1,2):0.3,
    (2,4):0.2,
    (4,5):0.1,
    (5,6):0.3,
    (6,7):0.2,
    (7,8):0.05,
    (8,9):0.05,
    (7,10):0.1,
    (2,11):0.2,
    (11,12):0.1,
    (12,13):0.3,
    (13,14):0.2,
    (14,15):0.05,
    (15,16):0.05,
    (14,17):0.1,
    (0,18):0.2,
    (18,19):0.4,
    (19,20):0.4,
    (20,21):0.1,
    (20, 21): 0.1,
    (20,32):0.05,
    (0,22):0.2,
    (22,23):0.4,
    (23,24):0.4,
    (24,25):0.1,
    (24,33):0.05
}

coco34_graph = nx.DiGraph()

coco34_graph.add_edges_from(COCO34_CONNECTIONS)


for edge,weight in COCO34_LENGTH.items():
    coco34_graph[edge[0]][edge[1]]['weight'] = weight

SMPL_KEYPOINTS = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

SMPL_CONNECTIONS = [
    # (0, 1), (0, 2), (0, 3),
    # (1, 4), (2, 5), (3, 6),
    # (4, 7), (5, 8), (6, 9),
    # (7, 10), (8, 11), (9, 12), (9, 13), (9, 14),
    # (14,15),
    # (15,16), (15,17),
    (0,1), (1,4), (4,7), (7,10),
    (0,2), (2,5), (5,8), (8,11),
    (0,3), (3,6), (6,9), (9,13),(9,14),(9,12),(12,15),
    (13,16), (16,18), (18,20), (20,22),
    (14,17), (17,19), (19,21), (21,23)

]

smpl24_graph = nx.DiGraph()

for i, keypoint_name in enumerate(SMPL_KEYPOINTS):
    smpl24_graph.add_node(i, label=keypoint_name)

smpl24_graph.add_edges_from(SMPL_CONNECTIONS)

for i, keypoint_name in enumerate(COCO34_KEYPOINTS):
    coco34_graph.add_node(i, label=keypoint_name)






skeleton_graph_dict = {"coco18": coco18_graph, "coco34": coco34_graph,"smpl24": smpl24_graph}