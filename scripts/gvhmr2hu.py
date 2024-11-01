from motion_convert.pipeline.smpl2hu_pipeline import  SMPL2HuPipeline
from motion_convert.pipeline.gvhmr2smpl_pipeline import GVHMR2SMPLPipeline
from motion_convert.pipeline.base_pipeline import PipelineArgs


if __name__ == '__main__':
    cam_out_path = 'motion_data/10_25/cam_out'
    smpl_out_path = 'motion_data/10_25/smpl'
    hu_out_path = 'motion_data/10_25/hu'

    gvhmr2hu_pipeline = GVHMR2SMPLPipeline(motion_dir=cam_out_path,
                                         save_dir=smpl_out_path, )
    gvhmr2hu_pipeline.run(debug=False)

    smpl2hu_pipeline = SMPL2HuPipeline(motion_dir=smpl_out_path,
                                      save_dir=hu_out_path)
    args = PipelineArgs(
        max_epoch=400,
        filter=True,
        fix_joints=True,
        joint_indices=[i for i in range(13, 33)],
        fix_ankles=True,
        zero_root=True,
        clip_angle=True,
        height_adjustment=False,
        move_to_ground=True,
        generate_mirror=True,
        save_info=True,
    )

    smpl2hu_pipeline.run(
        debug=False,
        **args
    )

