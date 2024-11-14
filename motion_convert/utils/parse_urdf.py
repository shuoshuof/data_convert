# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/12 15:17
@Auth ： shuoshuof
@File ：parse_urdf.py
@Project ：data_convert
"""
import os
import copy
from typing import List,Tuple
from collections import OrderedDict
import networkx as nx
import numpy as np

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonTree,SkeletonState
from poselib.poselib.core.rotation3d import *
from motion_convert.utils.torch_ext import to_numpy

from urdfpy import URDF
import trimesh

def parse_urdf(urdf_path):
    urdf_robot: URDF = URDF.load(urdf_path)
    # urdf_robot.show()
    fk_link = urdf_robot.link_fk()

    urdf_graph = copy.deepcopy(urdf_robot._G)

    link_parents = np.argmax(nx.adjacency_matrix(urdf_graph).todense(), axis=1).A1
    link_parents[0] = -1

    link_names = []
    link_mesh_file_names = []
    link_translations = []
    visual_origins = []
    # for (link,transform),joint in zip(fk_link.items(),urdf_robot.joints):
    for link, transform in fk_link.items():
        link_name = link.name
        link_translation = transform[:3, 3]
        # joint_translation = joint.origin[:3, 3]
        visual_origin = link.visuals[0].origin

        link_mesh_file_names.append(link.visuals[0].geometry.mesh.filename)
        link_names.append(link_name)
        link_translations.append(link_translation)
        # joint_translations.append(joint_translation)
        visual_origins.append(visual_origin)

    link_translations = np.array(link_translations)
    link_local_translations = np.zeros_like(link_translations)
    link_local_translations[1:] = link_translations[1:] - link_translations[link_parents[1:]]

    joint_names = []
    joint_translations = []
    joint_parent_names = []
    joint_name2joint_idx = {}
    for joint_idx, joint in enumerate(urdf_robot.joints):
        joint_name = joint.name
        joint_translation = joint.origin[:3, 3]
        joint_parent_names.append(joint.parent)
        joint_names.append(joint_name)
        joint_translations.append(joint_translation)

    joint_translations = np.array(joint_translations)
    # joint_local_translations = np.zeros_like(joint_translations)
    # joint_local_translations[1:] = joint_translations[1:]-joint_translations[joint_parent_names[1:]]

    # urdf_robot.show()

    # print(link_names)
    # print(link_parents)

    robot_sk_tree = SkeletonTree.from_dict(
        OrderedDict({'node_names': link_names,
                     'parent_indices': {'arr': np.array(link_parents), 'context': {'dtype': 'int64'}},
                     'local_translation': {'arr': np.array(link_local_translations), 'context': {'dtype': 'float32'}}})
    )

    robot_zero_pose = SkeletonState.zero_pose(robot_sk_tree)
    # from poselib.poselib.visualization.common import plot_skeleton_H
    # plot_skeleton_H([robot_zero_pose])
    return robot_zero_pose, link_mesh_file_names

def cal_urdf_mesh_bounding_boxes(urdf_path)->Tuple[List[trimesh.Trimesh],List[trimesh.primitives.Box]]:
    urdf_robot: URDF = URDF.load(urdf_path)
    links_trimesh = []
    for link in urdf_robot.links:
        trimesh_list = link.visuals[0].geometry.mesh.meshes
        assert len(trimesh_list) == 1
        links_trimesh.append(trimesh_list[0])

    meshes_bounding_boxes = []
    meshes_vertices = []
    for link_trimesh in links_trimesh:
        meshes_vertices.append(link_trimesh.vertices)
        meshes_bounding_boxes.append(link_trimesh.bounding_box)
    # print(meshes_vertices)
    # print(links_bounding_boxes)
    return links_trimesh,meshes_bounding_boxes

if __name__ == '__main__':
    # zero_pose, link_mesh_file_names = parse_urdf('asset/hu/hu_v5.urdf')
    cal_urdf_mesh_bounding_boxes('asset/hu/hu_v5.urdf')
