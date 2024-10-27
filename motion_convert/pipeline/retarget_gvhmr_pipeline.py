from smplx import SMPL
import torch
from tqdm import tqdm
from typing import Optional,Union
import numpy as np
import os
import multiprocessing
import joblib

from motion_convert.pipeline.base_pipeline import BasePipeline
from motion_convert.retarget_optimizer.smpl_retarget_optimizer import BaseSMPLRetargetOptimizer
from motion_convert.format_convert.smpl2isaac import convert2isaac

class RetargetGVHMR2SMPLPipeline(BasePipeline):
    def __init__(self,motion_dir:str,save_dir:str,smpl_model_path,num_processes:int=None):
        super().__init__(motion_dir,save_dir,num_processes)
        self.smpl_model_path = smpl_model_path
    def _read_data(self,**kwargs)->list:
        motion_paths = [os.path.join(self.motion_dir, f) for f in os.listdir(self.motion_dir) if
                              os.path.isfile(os.path.join(self.motion_dir, f))]
        return motion_paths

    def _split_data(self,data,**kwargs)->Union[np.ndarray, list]:
        return np.array_split(data,self.num_processes)

    def _rebuild_motion(self,motion_data)->torch.Tensor:
        body_pose = motion_data['smpl_params_global']['body_pose']
        betas = motion_data['smpl_params_global']['betas']
        global_orient = motion_data['smpl_params_global']['global_orient']
        transl = motion_data['smpl_params_global']['transl']

        add_body_pose = torch.zeros((body_pose.shape[0], 6))
        body_pose = torch.cat((global_orient, body_pose, add_body_pose), dim=1)

        body_pose_aa = body_pose[:, 3:].clone()
        global_orientation = body_pose[:, :3].clone()

        smpl = SMPL(self.smpl_model_path, gender="NEUTRAL")
        output = smpl(
            body_pose=body_pose_aa,
            global_orient=global_orientation,
            transl=transl,
        )

        rebuilt_motion = output.joints[:, :24, :]
        return rebuilt_motion
    def _coord_trans(self,rebuilt_motion)->torch.Tensor:
        new_motion = torch.zeros_like(rebuilt_motion)
        new_motion[..., 0] = -rebuilt_motion[..., 2]
        new_motion[..., 1] = -rebuilt_motion[..., 0]
        new_motion[..., 2] = rebuilt_motion[..., 1]
        return new_motion
    def _process_data(self,data_chunk,results,process_idx,**kwargs):
        smpl_retarget_optimizer = BaseSMPLRetargetOptimizer(smpl_model_path=self.smpl_model_path)
        for path in data_chunk:
            motion_data = torch.load(path)
            rebuilt_motion_data = self._rebuild_motion(motion_data)
            rebuilt_motion_data = self._coord_trans(rebuilt_motion_data)

            max_epoch = kwargs.get('max_epoch',2000)
            lr = kwargs.get('lr',1e-1)
            data = smpl_retarget_optimizer.train(
                rebuilt_motion_data,
                max_epoch=max_epoch,
                lr=lr,
                process_idx=process_idx
            )

            data = convert2isaac(data)
            data_dict = {}
            file_name = os.path.basename(path).split('.')[0]
            data_dict[file_name] = data

            with open(f'{self.save_dir}/{file_name}.pkl', 'wb') as f:
                joblib.dump(data_dict,f)


if __name__ == '__main__':
    gvhmr_pipeline = RetargetGVHMR2SMPLPipeline(motion_dir='test_data/pt_data1',
                                                save_dir='test_data/retargeted_data',
                                                smpl_model_path='asset/smpl')
    gvhmr_pipeline.run(debug=True,max_epoch=5000)
