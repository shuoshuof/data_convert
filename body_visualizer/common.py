from poselib.poselib.skeleton.skeleton3d import SkeletonMotion
from body_visualizer.visualizer import BodyVisualizer
def vis_motion(motion:SkeletonMotion,graph:str,static_frame=False):
    from motion_convert.robot_config.Hu import hu_graph
    from motion_convert.robot_config.VTRDYN import vtrdyn_graph
    graph_dict = {
        'hu':hu_graph,
        'vtrdyn':vtrdyn_graph
    }
    body_visualizer = BodyVisualizer(graph_dict[graph],static_frame=static_frame)
    for i,pos in enumerate(motion.global_translation):
        # print(i)
        body_visualizer.step(pos,close=False)