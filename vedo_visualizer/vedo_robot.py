# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/10 17:30
@Auth ： shuoshuof
@File ：vedo_robot.py
@Project ：data_convert
"""
import copy
from collections import OrderedDict
import networkx as nx
import numpy as np

from vedo import *

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonTree,SkeletonState
from urdfpy import URDF


class VedoRobot:
    def __init__(self,urdf_path,mesh_dir):
        self._parse_urdf(urdf_path)
    def _parse_urdf(self,urdf_path):
        urdf_robot:URDF = URDF.load(urdf_path)
        urdf_robot.show()
        fk_link = urdf_robot.link_fk()

        urdf_graph = copy.deepcopy(urdf_robot._G)

        link_parents = np.argmax(nx.adjacency_matrix(urdf_graph).todense(), axis=1).A1
        link_parents[0] = -1

        link_names = []
        link_translations = []

        for link,transform in fk_link.items():

            link_name = link.name
            # print(transform)
            link_translation = transform[:3,3]
            link_names.append(link_name)
            link_translations.append(link_translation)
        link_translations = np.array(link_translations)
        link_local_translations = np.zeros_like(link_translations)
        link_local_translations[1:] = link_translations[1:]-link_translations[link_parents[1:]]

        print(link_names)
        print(link_parents)

        robot_sk_tree = SkeletonTree.from_dict(
            OrderedDict({'node_names': link_names,
                         'parent_indices': {'arr': np.array(link_parents), 'context': {'dtype': 'int64'}},
                         'local_translation': {'arr': np.array(link_local_translations), 'context': {'dtype': 'float32'}}})
        )

        from poselib.poselib.visualization.common import plot_skeleton_H
        robot_zero_pose = SkeletonState.zero_pose(robot_sk_tree)
        plot_skeleton_H([robot_zero_pose])
    def _generate_robot(self):
        pass


if __name__ == '__main__':
    vedo_robot = VedoRobot(
        'asset/hu/hu_v5.urdf',
        'asset/hu/meshes_v5'
    )





