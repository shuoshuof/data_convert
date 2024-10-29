import copy
import joblib
import pickle

from poselib.poselib.visualization.common import *

from motion_convert.pipeline.base_pipeline import BasePipeline
from motion_convert.utils.transform3d import *
from motion_convert.retarget_optimizer.hu_retarget_optimizer import HuRetargetOptimizer
from motion_convert.utils.motion_process import *
from motion_convert.robot_config.Hu import *

class SMPL2HuPipeline(BasePipeline):
    def __init__(self, motion_dir: str, save_dir: str, num_processes: int = None):
        super().__init__(motion_dir, save_dir, num_processes)
        self.smpl_t_pose,self.hu_zero_pose,self.smpl_skeleton_tree,self.hu_skeleton_tree = self._load_asset()
    def _read_data(self, **kwargs) -> list:
        motion_paths = [os.path.join(self.motion_dir, f) for f in os.listdir(self.motion_dir) if
                              os.path.isfile(os.path.join(self.motion_dir, f))]
        return motion_paths
    def _split_data(self,data,**kwargs)->list:
        return np.array_split(data,self.num_processes)
    def _load_asset(self)->[SkeletonState, SkeletonState,SkeletonTree, SkeletonTree]:
        with open('asset/t_pose/smpl_t_pose.pkl','rb') as f:
            smpl_t_pose = pickle.load(f)
        with open('asset/zero_pose/hu_zero_pose.pkl','rb') as f:
            hu_zero_pose = pickle.load(f)
        with open('asset/smpl/smpl_skeleton_tree.pkl','rb') as f:
            smpl_skeleton_tree = pickle.load(f)
        hu_skeleton_tree = hu_zero_pose.skeleton_tree
        return smpl_t_pose,hu_zero_pose,smpl_skeleton_tree,hu_skeleton_tree
    def _rebuild_with_smpl_t_pose(self,motion:SkeletonMotion)->SkeletonMotion:
        motion_fps = motion.fps

        motion_global_translation = motion.global_translation
        motion_length, num_keypoint, _ = motion_global_translation.shape
        new_motion_global_rotation = motion.global_rotation
        new_motion_root_translation = motion.root_translation

        t_pose_local_translation = self.smpl_t_pose.local_translation.repeat(motion_length, 1, 1)
        parent_indices = self.smpl_t_pose.skeleton_tree.parent_indices

        rebuild_indices = [16,17,18,21,22,23]

        for joint_index in rebuild_indices:
            new_motion_global_rotation[:,parent_indices[joint_index],:] = \
                quat_between_two_vecs(t_pose_local_translation[:,joint_index],motion_global_translation[:,joint_index]-motion_global_translation[:,parent_indices[joint_index]])

        new_skeleton_state = SkeletonState.from_rotation_and_root_translation(
            self.smpl_t_pose.skeleton_tree,
            new_motion_global_rotation,
            new_motion_root_translation,
            is_local=False
        )
        new_motion = SkeletonMotion.from_skeleton_state(new_skeleton_state,fps=motion_fps)

        rebuild_error = torch.abs(new_motion.global_translation-motion_global_translation).max()
        print(f"Rebuild error :{rebuild_error:.5f}")

        return new_motion


    def _process_data(self,data_chunk,results,process_idx,debug,**kwargs):
        hu_retarget_optimizer = HuRetargetOptimizer(self.hu_skeleton_tree,kwargs.get('clip_angle',True))
        process_manager = MotionProcessManager()
        for path in data_chunk:
            with open(path,'rb') as f:
                motion_data = joblib.load(f)
            file_name = os.path.basename(path).split('.')[0]
            motion = motion_data[file_name]
            motion_fps = motion['fps']

            motion_global_rotation = motion['pose_quat_global']
            motion_root_translation = motion['root_trans_offset']

            smpl_motion = SkeletonState.from_rotation_and_root_translation(
                self.smpl_skeleton_tree,
                motion_global_rotation,
                motion_root_translation,
                is_local=False
            )

            # plot_skeleton_H([smpl_motion.zero_pose(self.smpl_skeleton_tree)])

            smpl_motion = SkeletonMotion.from_skeleton_state(smpl_motion,fps=motion_fps)
            # TODO： 腰部旋转很可能有问题
            rebuilt_motion = self._rebuild_with_smpl_t_pose(smpl_motion)

            target_motion = copy.deepcopy(rebuilt_motion).retarget_to_by_tpose(
                joint_mapping=SMPL2HU_JOINT_MAPPING,
                source_tpose=self.smpl_t_pose,
                target_tpose=self.hu_zero_pose,
                rotation_to_target_skeleton=torch.Tensor([0,0,0,1]),
                scale_to_target_skeleton=1.,
            )

            max_epoch = kwargs.get('max_epoch',500)
            lr = kwargs.get('lr',1e-1)
            retargeted_motion = hu_retarget_optimizer.train(
                motion_data=target_motion,
                max_epoch=max_epoch,
                lr=lr,
                process_idx=process_idx
            )

            result_motion = process_manager.process_motion(retargeted_motion,**kwargs)

            motion_dict = {}
            motion_dict['pose_quat_global'] = result_motion.global_rotation.numpy()
            motion_dict['pose_quat'] = result_motion.local_rotation.numpy()
            motion_dict['trans_orig'] = None
            motion_dict['root_trans_offset'] = result_motion.root_translation.numpy()
            motion_dict['target'] = "hu"
            motion_dict['pose_aa'] = None
            motion_dict['fps'] = motion_fps
            motion_dict['project_loss'] = 1e-5

            save_dict  = {file_name:motion_dict}
            with open(f'{self.save_dir}/{file_name}.pkl', 'wb') as f:
                joblib.dump(save_dict,f)
            with open(f'{self.save_dir}/{file_name}_motion.pkl', 'wb') as f:
                joblib.dump(result_motion,f)

            # plot_skeleton_H([target_motion,result_motion])

if __name__ == '__main__':

    smpl2hu_pipeline = SMPL2HuPipeline(motion_dir='motion_data/10_24_1/smpl',
                                      save_dir='motion_data/10_24_1/hu')
    smpl2hu_pipeline.run(
        debug=False,
        max_epoch=400,
        fix_root=False,
        move_to_ground=False,
        filter=False,
        clip_angle=True,
        height_adjustment=True,
    )