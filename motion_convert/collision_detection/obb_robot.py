# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/9 17:50
@Auth ： shuoshuof
@File ：obb_robot.py
@Project ：data_convert
"""

from typing import Union,List
from collections import OrderedDict
import time
import copy

import torch

import trimesh

from poselib.poselib.core.rotation3d import quat_rotate
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion,SkeletonTree,SkeletonState

from motion_convert.utils.torch_ext import to_torch

class OBBRobot:
    def __init__(self,num_links,device='cuda'):
        self.device = device
        self.num_links = num_links
    @staticmethod
    def load_robot(urdf_path:str):
        from motion_convert.utils.parse_urdf import parse_urdf,cal_urdf_mesh_bounding_boxes

        robot_zero_pose,_ = parse_urdf(urdf_path)
        links_trimesh,meshes_bounding_boxes = cal_urdf_mesh_bounding_boxes(urdf_path)
        return robot_zero_pose,links_trimesh,meshes_bounding_boxes

    @classmethod
    def from_dict(cls,obb_dict:OrderedDict,device='cuda'):
        obb_robot = cls(len(obb_dict),device)
        obb_robot._init_obbs_from_dict(obb_dict)
        return obb_robot

    def _init_obbs_from_dict(self,obb_dict):
        initial_axes = []
        extents = []
        initial_offsets = []
        global_rotations = []
        global_translations = []
        obb_link_indices = []
        for link_idx, (obb_name, obb_data) in enumerate(obb_dict.items()):
            initial_axes.append(obb_data['initial_axes'])
            extents.append(obb_data['extents'])
            initial_offsets.append(obb_data['initial_offsets'])
            global_rotations.append(obb_data['global_rotation'])
            global_translations.append(obb_data['global_translation'])
            obb_link_indices.append(link_idx)

        self._initial_axes = torch.stack(initial_axes, dim=0).to(self.device)
        self._extents = torch.stack(extents, dim=0).to(self.device)
        self._initial_offsets = torch.stack(initial_offsets, dim=0).to(self.device)
        self._global_rotations = torch.stack(global_rotations, dim=0).to(self.device)
        self._global_translations = torch.stack(global_translations, dim=0).to(self.device)
        self._obb_link_indices = torch.tensor(obb_link_indices, dtype=torch.int64).to(self.device)

        self._num_obbs = len(self._obb_link_indices)

        self._collision_mask_mat = torch.ones((self.num_obbs,self.num_obbs),dtype=torch.int32).to(self.device)

        self._offsets = self._cal_offsets()
        self._axes = self._cal_axes()
        self._vertices = self._cal_vertices()

        assert self._axes.shape == (self.num_obbs,3,3)
        assert self._vertices.shape == (self.num_obbs,8,3)

    @classmethod
    def from_urdf(cls,urdf_path,divide=False,device='cuda',):
        # TODO：remove the dependency of poselib
        robot_zero_pose, links_trimesh, meshes_bounding_boxes = cls.load_robot(urdf_path)
        obb_robot = cls(len(links_trimesh),device)
        obb_robot._init_obbs_from_urdf(robot_zero_pose, links_trimesh, meshes_bounding_boxes, divide)
        return obb_robot

    def _init_obbs_from_urdf(self, robot_zero_pose:SkeletonState, links_trimesh:List[trimesh.Trimesh], meshes_bounding_boxes:List[trimesh.primitives.Box], divide):

        if divide:
            self._cal_divided_obbs(robot_zero_pose,links_trimesh)
        else:
            self._cal_simple_obbs(robot_zero_pose,meshes_bounding_boxes)

        self._num_obbs = len(self._obb_link_indices)

        self._collision_mask_mat = self._cal_collision_mask_mat(robot_zero_pose.skeleton_tree.parent_indices)

        self._offsets = self._cal_offsets()
        self._axes = self._cal_axes()
        self._vertices = self._cal_vertices()

        assert self._axes.shape == (self.num_obbs,3,3)
        assert self._vertices.shape == (self.num_obbs,8,3)

    def _cal_collision_mask_mat(self,parent_indices):
        mask = torch.ones((self.num_obbs,self.num_obbs),dtype=torch.int32).to(self.device)
        # filter out the collision between the parent link and child link

        for obb_idx in range(self.num_obbs):
            obb_link_idx = self._obb_link_indices[obb_idx]
            obb_link_parent_idx = parent_indices[obb_link_idx]
            if obb_link_parent_idx != -1:
                mask[obb_idx,torch.argwhere(self._obb_link_indices==obb_link_parent_idx)] = 0
                mask[torch.argwhere(self._obb_link_indices==obb_link_parent_idx),obb_idx] = 0

        # the collision of obbs which have the same link index should not be considered
        for obb_idx in range(self.num_obbs):
            obb_link_idx = self._obb_link_indices[obb_idx]
            mask_idx = torch.argwhere(self._obb_link_indices==obb_link_idx)
            mask[obb_idx,mask_idx] = 0
            mask[mask_idx,obb_idx] = 0
        return mask
    def _cal_simple_obbs(self,robot_zero_pose,meshes_bounding_boxes):
        initial_axes = []
        extents = []
        initial_offsets = []
        # TODO: add rotation offsets
        initial_rotations_offsets = []
        global_rotations = []
        global_translations = []
        obb_link_indices = []
        for link_idx, box in enumerate(meshes_bounding_boxes):
            initial_axes.append(torch.eye(3))
            extents.append(to_torch(copy.deepcopy(box.extents))/2)
            initial_offsets.append(to_torch(copy.deepcopy(box.transform[:3, 3])))
            global_rotations.append(torch.tensor([0,0,0,1]))
            global_translations.append(robot_zero_pose.global_translation[link_idx])
            obb_link_indices.append(link_idx)

        self._initial_axes = torch.stack(initial_axes,dim=0).to(self.device)
        self._extents = torch.stack(extents,dim=0).to(self.device)
        self._initial_offsets = torch.stack(initial_offsets,dim=0).to(self.device)
        self._global_rotations = torch.stack(global_rotations,dim=0).to(self.device)
        self._global_translations = torch.stack(global_translations,dim=0).to(self.device)
        self._obb_link_indices = torch.tensor(obb_link_indices,dtype=torch.int64).to(self.device)

    def _cal_divided_obbs(self,robot_zero_pose,links_trimesh:List[trimesh.Trimesh]):
        r"""
        divide the mesh bounding box into 8 sub-boxes, and use the grid search to find the best division
        :param robot_zero_pose:
        :param links_trimesh:
        :return:
        """
        initial_axes = []
        extents = []
        initial_offsets = []
        # TODO: add rotation offsets
        initial_rotations_offsets = []
        global_rotations = []
        global_translations = []
        obb_link_indices = []

        start = time.time()
        search_step = 0.02
        for link_idx, mesh in enumerate(links_trimesh):
            # some resolutions of mesh are too small, which may cause the boxes can't contain all vertices
            mesh = mesh.subdivide_loop(1)
            mesh_bbox = mesh.bounding_box
            mesh_volume = torch.tensor(mesh.volume).to(self.device)
            vertices = to_torch(mesh.vertices).to(self.device)

            min_bounds,max_bounds = to_torch(copy.deepcopy(mesh.bounds)).to(self.device)
            x_bounds = torch.linspace(min_bounds[0],max_bounds[0],int((max_bounds[0]-min_bounds[0])/search_step))
            y_bounds = torch.linspace(min_bounds[1],max_bounds[1],int((max_bounds[1]-min_bounds[1])/search_step))
            z_bounds = torch.linspace(min_bounds[2],max_bounds[2],int((max_bounds[2]-min_bounds[2])/search_step))

            grid_x, grid_y, grid_z = torch.meshgrid(x_bounds,y_bounds,z_bounds,indexing='ij')
            # (length,width,height,3)
            grid = torch.concatenate([grid_x.unsqueeze(-1),grid_y.unsqueeze(-1),grid_z.unsqueeze(-1)],dim=-1).to(self.device)
            assert grid.shape == (len(x_bounds),len(y_bounds),len(z_bounds),3)

            signs = torch.tensor([-1,1],dtype=torch.float32).to(self.device)
            # (8,3)
            signs = torch.cartesian_prod(signs,signs,signs)
            # (length,width,height,8,num_vertices) <-
            # (length,width,height,num_vertices,3).unsqueeze(-3) (8,3).unsqueeze(1)
            mask = torch.all((vertices.unsqueeze(0) - grid.unsqueeze(-2)).unsqueeze(-3) * signs.unsqueeze(1) >= 0,dim=-1)
            assert mask.shape == (len(x_bounds),len(y_bounds),len(z_bounds),8,len(vertices))

            fill_grid = grid.clone().view(len(x_bounds),len(y_bounds),len(z_bounds),1,1,3)

            masked_vertices = torch.where(mask.unsqueeze(-1),vertices.unsqueeze(0),fill_grid)
            boxes_bounds_max = masked_vertices.max(dim=-2).values
            boxes_bounds_min = masked_vertices.min(dim=-2).values

            assert boxes_bounds_max.shape==boxes_bounds_min.shape==(len(x_bounds),len(y_bounds),len(z_bounds),8,3)

            volume_each_division = (boxes_bounds_max - boxes_bounds_min).prod(dim=-1)
            assert torch.all(volume_each_division>=0)

            volume_each_division = volume_each_division.sum(dim=-1)
            assert volume_each_division.shape == (len(x_bounds),len(y_bounds),len(z_bounds))

            assert mesh.volume<= volume_each_division.min()
            division_idx = torch.argmin((volume_each_division-mesh_volume).abs())
            division_idx = torch.unravel_index(division_idx,volume_each_division.shape)

            boxes_bounds_max = boxes_bounds_max[division_idx].view(8,3)
            boxes_bounds_min = boxes_bounds_min[division_idx].view(8,3)

            # print(f"-----------link: {link_idx}-----------------")
            for i in range(8):
                # filter out zero volume obbs
                if torch.all(boxes_bounds_max[i]-boxes_bounds_min[i]<=1e-6):
                    continue
                initial_axes.append(torch.eye(3))
                extents.append((boxes_bounds_max[i]-boxes_bounds_min[i])/2)
                initial_offsets.append((boxes_bounds_max[i]+boxes_bounds_min[i])/2)
                global_rotations.append(torch.tensor([0,0,0,1]))
                global_translations.append(robot_zero_pose.global_translation[link_idx])
                obb_link_indices.append(link_idx)
                # print(f"offset: {initial_offsets[-1]}")
                # print(f"grid bounds min: {boxes_bounds_min[i]}")
                # print(f"grid bounds max: {boxes_bounds_max[i]}")

        end = time.time()
        torch.cuda.empty_cache()
        print(f"divide time cost: {end-start}, got {len(obb_link_indices)} obbs")

        self._initial_axes = torch.stack(initial_axes,dim=0).to(self.device)
        self._extents = torch.stack(extents,dim=0).to(self.device)
        self._initial_offsets = torch.stack(initial_offsets,dim=0).to(self.device)
        self._global_rotations = torch.stack(global_rotations,dim=0).to(self.device)
        self._global_translations = torch.stack(global_translations,dim=0).to(self.device)
        self._obb_link_indices = torch.tensor(obb_link_indices,dtype=torch.int64).to(self.device)

    @property
    def num_obbs(self):
        return self._num_obbs
    @property
    def axes(self):
        return self._axes.clone()
    @property
    def extents(self):
        return self._extents.clone()
    @property
    def offsets(self):
        return self._offsets.clone()
    @property
    def global_rotations(self):
        return self._global_rotations.clone()
    @property
    def global_translations(self):
        return self._global_translations.clone()
    @property
    def vertices(self):
        return self._vertices.clone()
    @property
    def center_pos(self):
        return self.global_translations + self.offsets
    @property
    def obbs_collision_mask_mat(self):
        return self._collision_mask_mat.clone()

    def _cal_axes(self):
        return quat_rotate(self.global_rotations.unsqueeze(1),self._initial_axes.clone())

    def _cal_vertices(self):
        signs = torch.tensor([-1,1],dtype=torch.float32).to(self.device)
        vertices = torch.cartesian_prod(signs,signs,signs)@(self.extents.unsqueeze(-1)*self.axes)
        # the axes have been rotated, so vertices don't need to
        return vertices.clone() + self.center_pos.unsqueeze(1)


    def _cal_offsets(self):
        return quat_rotate(self.global_rotations, self._initial_offsets.clone())

    def update_transform(self, global_translations=None, global_rotations=None, from_link_transform=False):
        if global_translations is not None:
            assert global_translations.shape == (self.num_obbs, 3)
            self._global_translations = global_translations
        if global_rotations is not None:
            assert global_rotations.shape == (self.num_obbs, 4)
            self._global_rotations = global_rotations
        # TODO: check _cal_offsets
        # if from_link_transform:
        #     # offset should be rotated
        #     self._offsets = self._cal_offsets()
        # offset should be rotated
        self._offsets = self._cal_offsets()
        self._axes = self._cal_axes()
        self._vertices = self._cal_vertices()

class OBBRobotCollisionDetector:
    def __init__(
            self,
            obb_robot:OBBRobot,
            link_collision_mask:torch.Tensor=None,
            use_zero_pose_mask=True,
            device='cuda'
    ):
        self._obb_robot = obb_robot
        self._num_obbs = self._obb_robot.num_obbs
        self.device = device
        # the collision of two near links from a robot may not be considered as collision
        self.collision_mask = obb_robot.obbs_collision_mask_mat
        # filter out the collision of two near links from a robot
        if use_zero_pose_mask:
            self.update_obbs_transform(None,None,from_link_transform=False)
            zero_pose_mask = torch.logical_not(self.check_collision(return_obbs_collisions=True))
            self.collision_mask *= zero_pose_mask
        if link_collision_mask is not None:
            assert link_collision_mask.shape == (self.obb_robot_num_links,self.obb_robot_num_links)
            self.link_collision_mask = link_collision_mask.to(self.device)
        else:
            self.link_collision_mask = torch.ones((self.obb_robot_num_links,self.obb_robot_num_links)).bool().to(self.device)
        if self.num_obbs == self.obb_robot_num_links:
            self.collision_mask *=self.link_collision_mask
    @property
    def num_obbs(self):
        return self._num_obbs
    @property
    def obb_robot(self):
        return self._obb_robot
    def obb_robot_global_translation(self)->torch.Tensor:
        return self.obb_robot.global_translations
    def obb_robot_global_rotation(self)->torch.Tensor:
        return self.obb_robot.global_rotations
    def obb_robot_center_pos(self)->torch.Tensor:
        return self.obb_robot.center_pos
    def obb_robot_offsets(self)->torch.Tensor:
        return self.obb_robot.offsets
    def obb_robot_vertices(self)->torch.Tensor:
        return self.obb_robot.vertices
    def obb_robot_axes(self)->torch.Tensor:
        return self.obb_robot.axes
    def obb_robot_extents(self)->torch.Tensor:
        return self.obb_robot.extents
    @property
    def obb_robot_obb_link_indices(self)->torch.Tensor:
        return self.obb_robot._obb_link_indices.clone()
    @property
    def obb_robot_num_links(self)->int:
        return self.obb_robot.num_links
    def _cal_obbs_separating_axes_tensor(self):
        r"""
        main_axes: (num_obbs,num_obbs,6,3)
        cross_axes: (num_obbs,num_obbs,9,3)
        :return: splitting axes tensor with shape (num_obbs,num_obbs,15,3)
        """
        main_axes = torch.concatenate(
            [self.obb_robot_axes().unsqueeze(1).repeat(1, self.num_obbs, 1, 1),
             self.obb_robot_axes().unsqueeze(0).repeat(self.num_obbs, 1, 1, 1)],dim=-2)
        assert main_axes.shape == (self.num_obbs,self.num_obbs,6,3)

        # obbs_axes: (num_obbs,3,3) -> (num_obbs,1,3,3) -> (num_obbs,num_obbs,3,3) -> (num_obbs,num_obbs,3,1,3)
        edge1 = self.obb_robot_axes().unsqueeze(1).repeat(1, self.num_obbs, 1, 1).unsqueeze(-2)
        # obbs_axes: (num_obbs,3,3) -> (1,num_obbs,3,3) -> (num_obbs,num_obbs,3,3) -> (num_obbs,num_obbs,1,3,3)
        edge2 = self.obb_robot_axes().unsqueeze(0).repeat(self.num_obbs, 1, 1, 1).unsqueeze(-3)

        cross_axes = torch.cross(edge1,edge2,dim=-1).view(self.num_obbs,self.num_obbs,9,3)
        # the cross product of two parallel edges is zero, set a default vector
        default_vector = torch.tensor([1, 0, 0],dtype=torch.float32,device=self.device)
        cross_axes = torch.where(cross_axes.norm(dim=-1,keepdim=True)<=1e-6,default_vector,cross_axes)
        assert cross_axes.shape == (self.num_obbs,self.num_obbs,9,3)

        separating_axes = torch.cat([main_axes,cross_axes],dim=-2)
        assert separating_axes.shape == (self.num_obbs,self.num_obbs,15,3)

        return separating_axes
    def _cal_obbs_centers_tensor(self):
        r"""
        return the diff between each pair of obbs's centers with shape (num_obbs,num_obbs,2,3)
        """
        obbs_centers =  torch.concatenate(
            [self.obb_robot_center_pos().unsqueeze(1).repeat(1, self.num_obbs, 1).unsqueeze(-2),
             self.obb_robot_center_pos().unsqueeze(0).repeat(self.num_obbs, 1, 1).unsqueeze(-2)],dim=-2)
        assert obbs_centers.shape == (self.num_obbs,self.num_obbs,2,3)
        return obbs_centers

    def _cal_obbs_centers_diff_tensor(self, obbs_centers_tensor):
        return (obbs_centers_tensor.clone()[..., [0], :]- obbs_centers_tensor.clone()[..., [1], :]).abs()

    def _cal_obbs_centers_diff_proj_tensor(self,obb_separating_axes_tensor,obbs_centers_diff_tensor):
        return torch.sum(obb_separating_axes_tensor*obbs_centers_diff_tensor,dim=-1).abs()

    def _cal_obbs_vertices_tensor(self):
        r"""
        :return: vertices tensor with shape (num_obbs,num_obbs,2,8,3)
        """
        obbs_vertices = torch.concatenate(
            [self.obb_robot_vertices().unsqueeze(1).repeat(1, self.num_obbs, 1, 1).unsqueeze(-3),
             self.obb_robot_vertices().unsqueeze(0).repeat(self.num_obbs, 1, 1, 1).unsqueeze(-3)],dim=-3)
        assert obbs_vertices.shape == (self.num_obbs,self.num_obbs,2,8,3)
        return obbs_vertices
    def _cal_obbs_vertices_vec_proj_dist_tensor(self, obbs_centers_tensor,obb_separating_axes_tensor):
        r"""
        :param obbs_centers_tensor: shape (num_obbs,num_obbs,2,3)
        :param obb_separating_axes_tensor: shape (num_obbs,num_obbs,15,3)
        :return: projection distances tensor with shape (num_obbs,num_obbs,2,15)
        """
        obbs_vertices_tensor = self._cal_obbs_vertices_tensor()
        obbs_vertices_vec_tensor = obbs_vertices_tensor - obbs_centers_tensor.unsqueeze(-2)
        # [ (num_obbs,num_obbs,2,8,3)->(num_obbs,num_obbs,2,8,1,3)] [(num_obbs,num_obbs,15,3)->(num_obbs,num_obbs,1,1,15,3)]
        # -> (num_obbs,num_obbs,2,8,15)
        obbs_vertices_vec_proj = torch.sum(obbs_vertices_vec_tensor.unsqueeze(-2)*obb_separating_axes_tensor.unsqueeze(2).unsqueeze(-3),dim=-1)
        assert obbs_vertices_vec_proj.shape == (self.num_obbs,self.num_obbs,2,8,15)

        obbs_vertices_vec_proj_dist_tensor = torch.abs(obbs_vertices_vec_proj)
        obbs_vertices_vec_proj_dist_tensor = torch.max(obbs_vertices_vec_proj_dist_tensor,dim=-2).values
        assert obbs_vertices_vec_proj_dist_tensor.shape == (self.num_obbs,self.num_obbs,2,15)

        return obbs_vertices_vec_proj_dist_tensor
    def obbs_collision_mat_to_links_collision_mat(self, obbs_collision_mat):
        r"""
        :param obbs_collision_mat: shape (num_obbs,num_obbs)
        :return: a links collision matrix with shape (num_links,num_links)
        """
        # # TODO: fix this bug
        # # (num_obbs,num_links)
        # row_indices = self.obb_robot_obb_link_indices.unsqueeze(1).expand(-1, self.obb_robot_num_links)
        # # (num_links,num_obbs)
        # col_indices = self.obb_robot_obb_link_indices.unsqueeze(0).expand(self.obb_robot_num_links, -1)
        #
        # links_collision_mat = torch.zeros((self.obb_robot_num_links, self.obb_robot_num_links),dtype=torch.int64, device=self.device)
        #
        # links_collision_mat.scatter_add_(0,row_indices,obbs_collision_mat)
        # links_collision_mat.scatter_add_(1,col_indices,obbs_collision_mat)

        # links_collision_mat = torch.zeros((self.obb_robot_num_links, self.obb_robot_num_links), dtype=torch.int64,device=self.device)
        # for r_link_idx in range(self.obb_robot_num_links):
        #     for w_link_idx in range(self.obb_robot_num_links):
        #         r_indices = torch.argwhere(self.obb_robot_obb_link_indices==torch.Tensor([r_link_idx]).to(self.device))
        #         w_indices = torch.argwhere(self.obb_robot_obb_link_indices==torch.Tensor([w_link_idx]).to(self.device))
        #         links_collision_mat[r_link_idx][w_link_idx] = obbs_collision_mat[r_indices,w_indices].min()

        row_indices = self.obb_robot_obb_link_indices.unsqueeze(1).expand(-1, self.num_obbs)
        col_indices = self.obb_robot_obb_link_indices.unsqueeze(0).expand(self.obb_robot_num_links, -1)

        tmp_links_collision_mat = torch.zeros((self.obb_robot_num_links, self.num_obbs),dtype=torch.int64, device=self.device)

        tmp_links_collision_mat.scatter_add_(0,row_indices,obbs_collision_mat)

        links_collision_mat = torch.zeros((self.obb_robot_num_links, self.obb_robot_num_links),dtype=torch.int64, device=self.device)
        links_collision_mat.scatter_add_(1,col_indices,tmp_links_collision_mat)

        return (links_collision_mat*self.link_collision_mask).to(torch.bool)

    def check_collision(self,return_obbs_collisions:bool=False):
        r"""
              separating axes              vertices                centers diff             projection
        (num_obbs,num_obbs,15,3), (num_obbs,num_obbs,2,8,3), (num_obbs,num_obbs,1,3)   (num_obbs,num_obbs,15)
        :param return_obbs_collisions: bool whether to return the obbs' collision matrix,
                                       default to return the link collision matrix
        :return: a collision matrix with shape (num_obbs,num_obbs)
        """
        # TODO: filter the collision between two obbs whose distance are far

        obb_separating_axes_tensor = self._cal_obbs_separating_axes_tensor()

        obbs_centers_tensor = self._cal_obbs_centers_tensor()
        obbs_centers_diff_tensor = self._cal_obbs_centers_diff_tensor(obbs_centers_tensor)

        # (num_obbs,num_obbs,15,3) (num_obbs,num_obbs,1,3) -> (num_obbs,num_obbs,15)
        obbs_centers_diff_proj_tensor = self._cal_obbs_centers_diff_proj_tensor(obb_separating_axes_tensor,obbs_centers_diff_tensor)

        assert obbs_centers_diff_proj_tensor.shape == (self.num_obbs,self.num_obbs,15)

        obbs_vertices_vec_proj_dist_tensor = self._cal_obbs_vertices_vec_proj_dist_tensor(obbs_centers_tensor,obb_separating_axes_tensor)
        obbs_vertices_vec_proj_dist_sum_tensor = torch.sum(obbs_vertices_vec_proj_dist_tensor,dim=-2)
        assert obbs_vertices_vec_proj_dist_sum_tensor.shape == (self.num_obbs,self.num_obbs,15)

        collision_mat = torch.where(obbs_centers_diff_proj_tensor>obbs_vertices_vec_proj_dist_sum_tensor,0,1)
        collision_mat = torch.min(collision_mat, dim=-1).values

        assert torch.allclose(collision_mat, collision_mat.transpose(0,1),atol=1e-6), 'collision_mat is not symmetric'
        assert collision_mat.shape == (self.num_obbs,self.num_obbs)

        collision_mat = collision_mat*self.collision_mask


        if not return_obbs_collisions:
            return self.obbs_collision_mat_to_links_collision_mat(collision_mat)

        return collision_mat.to(torch.bool)
    def update_obbs_transform(self, global_translations, global_rotations, from_link_transform=False):
        obb_global_translations=None
        obb_global_rotations=None

        if global_translations is not None and from_link_transform:
            obb_global_translations = global_translations.to(self.device)[self.obb_robot_obb_link_indices]
        if global_rotations is not None and from_link_transform:
            obb_global_rotations = global_rotations.to(self.device)[self.obb_robot_obb_link_indices]

        self._obb_robot.update_transform(
            global_translations=obb_global_translations,
            global_rotations=obb_global_rotations,
            from_link_transform=from_link_transform
        )


if __name__ == '__main__':

    obb_robot = OBBRobot.from_urdf(urdf_path='asset/hu/hu_v5.urdf',divide=True)
    obb_detector = OBBRobotCollisionDetector(obb_robot=obb_robot)

    import time
    for i in range(10000):
        start = time.time()
        obb_detector.update_obbs_transform(
            global_translations=torch.randn(obb_detector.num_obbs,3),
            global_rotations=torch.randn(obb_detector.num_obbs, 4),
        )
        # print(obb_detector.obb_robot_global_rotation())
        obb_detector.check_collision(return_obbs_collisions=False)
        end = time.time()

        print(f"cal_obb: {(end - start) }")

    # from vedo_visualizer.vedo_robot import VedoOBBRobot
    # from vedo_visualizer.common import vis_sk_motion
    # import pickle
    #
    # # vedo_obb_robot = VedoOBBRobot.from_obb_detector(obb_detector,vis_links_indices=[0])
    # # vedo_obb_robot.show()
    # with open('motion_data/11_7_walk/hu_motion/walk_small_step1_11_07_22_mirror_motion.pkl','rb') as f:
    #     motion = pickle.load(f)
    #
    # # vis_sk_motion([copy.deepcopy(motion)],divide=False)
    # vis_sk_motion([copy.deepcopy(motion)],divide=True,vis_links_indices=[14])
