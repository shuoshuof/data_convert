from motion_convert.pipeline.smpl2hu_pipeline import  SMPL2HuPipeline
from motion_convert.pipeline.gvhmr2smpl_pipeline import GVHMR2SMPLPipeline


if __name__ == '__main__':
    cam_out_path = 'motion_data/10_25/cam_out'
    smpl_out_path = 'motion_data/10_25/smpl'
    hu_out_path = 'motion_data/10_25/hu'

    gvhmr2hu_pipeline = GVHMR2SMPLPipeline(motion_dir=cam_out_path,
                                         save_dir=smpl_out_path, )
    gvhmr2hu_pipeline.run(debug=False)

    smpl2hu_pipeline = SMPL2HuPipeline(motion_dir=smpl_out_path,
                                      save_dir=hu_out_path)
    smpl2hu_pipeline.run(
        debug=False,
        max_epoch=400,
        fix_root=False,
        move_to_ground=True,
        filter=False,
        clip_angle=True
    )

