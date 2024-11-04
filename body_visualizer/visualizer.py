import matplotlib.pyplot as plt
import numpy as np
from body_visualizer.skeleton_graph import *
import time

class BodyVisualizer:
    def __init__(self,skeleton_graph=coco34_graph,static_frame=True):
        self.fig = plt.figure(figsize=(5,5))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.static_frame = static_frame

        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+1500+0")
        self.ax.view_init(elev=0, azim=0)

        # self.skeleton_graph = skeleton_graph_dict[skeleton_format]
        self.skeleton_graph = skeleton_graph
        self.motion_list = []
    def set_axis(self,root_pos):
        if self.static_frame:
            self.ax.set_xlim([-1,1])
            self.ax.set_ylim([-1,1])
            self.ax.set_zlim([-1,1])
        else:
            self.ax.set_xlim([root_pos[0] - 1, root_pos[0] + 1])
            self.ax.set_ylim([root_pos[1] - 1, root_pos[1] + 1])
            self.ax.set_zlim([root_pos[2] - 1, root_pos[2] + 1])

        # 设置坐标轴标签
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')


    def update_plot(self,keypoint):
        self.ax.cla()
        self.set_axis(root_pos=keypoint[0])

        if keypoint.size != 0:
            valid_points = keypoint[~np.isnan(keypoint).any(axis=1)]

            if valid_points.size != 0:
                x = valid_points[:, 0]
                y = valid_points[:, 1]
                z = valid_points[:, 2]

                self.ax.scatter(x, y, z, c='r', marker='o')
            self.skeleton_dfs(0,keypoint)

        for idx in range(len(keypoint)):
            x,y, z = keypoint[idx]
            self.ax.text(x,
                         y,
                         z,
                         f'{idx}',
                         color='black',
                         fontsize=10,
                         verticalalignment='bottom',
                         horizontalalignment='right')
    def skeleton_dfs(self,node,keypoint):
        for child_node in  self.skeleton_graph.successors(node):
            start = keypoint[node]
            end = keypoint[child_node]
            if np.isnan(start).any() or np.isnan(end).any():
                continue
            self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'b-')
            self.skeleton_dfs(child_node,keypoint)
    def step(self,keypoint,close=False):

        self.update_plot(keypoint)
        plt.draw()
        plt.pause(0.01)
        if close:
            plt.close()




if __name__ == "__main__":
    import pickle
    import joblib
    from poselib.poselib.skeleton.skeleton3d import SkeletonMotion
    from motion_convert.robot_config.Hu import hu_graph
    # with open('../standard_motion.pkl', 'rb') as f:
    #     data = pickle.load(f)
    motion_new: SkeletonMotion = joblib.load(f"motion_data/10_25/hu/10_25_walking_1_motion.pkl")
    visualizer = BodyVisualizer(hu_graph)
    for pos in motion_new.global_translation:
        visualizer.step(pos, close=False)


