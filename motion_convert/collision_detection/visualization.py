import matplotlib.pyplot as plt
import torch
from vedo import *
from typing import List
from motion_convert.collision_detection.obb import OBB,OBBCollisionDetector
from poselib.poselib.core.rotation3d import *
import time
class ObbVisualizer:
    def __init__(self, obbs:List[OBB]):
        self.obbs_boxes = [ObbBox(obb) for obb in obbs]

        world_frame =  Box(pos=[0,0,0],length=2,width=2,height=2).wireframe()

        self.plt = Plotter()
        self.plt.add(world_frame)
    def update_obbs_boxes(self,obbs:List[OBB]):
        for i,obb in enumerate(obbs):
            self.obbs_boxes[i].update(obb)

    def render(self):
        self.plt.show(*self.obbs_boxes,axes=1,viewup='z',interactive=False)




class ObbBox(Box):
    def __init__(self, obb:OBB,trail=10):

        pos = torch_to_numpy(obb.center)
        length, width, height = 2*torch_to_numpy(obb.extents)
        super().__init__(pos,length,width,height)
        self.update(obb)

        if trail > 0:
            self.add_trail(trail)

    def update(self,obb:OBB):
        obb_rotation_angle,obb_rotation_axis = quat_to_angle_axis(obb.global_rotation)
        obb_rotation_axis = torch_to_numpy(obb_rotation_axis)
        obb_rotation_angle = torch_to_numpy(obb_rotation_angle)
        self.rotate(angle=obb_rotation_angle, axis=obb_rotation_axis)
        self.pos(*obb.global_translation)




def torch_to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().detach().numpy()
    return t

if __name__ == "__main__":

    obb1 = OBB(
        center=torch.tensor([0,0,0]),
        axes=torch.eye(3),
        extents=torch.tensor([1e-2,1e-2,1e-2]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor((0,0,0)),
    )

    obb2 = OBB(
        center=torch.tensor([1,0,0]),
        axes=torch.eye(3),
        extents=torch.tensor([1e-2,1e-2,1e-2]),
        global_rotation=torch.tensor([0,0,0,1]),
        global_translation=torch.tensor((0,0,0)),
    )

    obb_detector = OBBCollisionDetector([obb1,obb2]*20)
    obbv = ObbVisualizer(obb_detector.get_obbs())



    for i in range(1000):

        obb_detector.update_obbs_transform(
            global_rotations=torch.tensor([[0,0,0,1],[0,0,0,1]]),
            global_translations=torch.tensor([[i/1000,0,0],[-i/1000,0,0]]),
        )
        obbv.update_obbs_boxes(obb_detector.get_obbs())
        obbv.render()

