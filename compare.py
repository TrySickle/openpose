import numpy as np
import matplotlib.pyplot as plt


golf_openpose_2d = np.load('data_2d_detections_golf_openpose.npz')
golf_openpose_3d = np.load('out_3D_vp3d_detections_golf_openpose.npz')
golf_detectron_2d = np.load('data_2d_detections_golf_detectron.npz')
golf_detectron_3d = np.load('out_3D_vp3d_detections_golf.npz')

kp1 = golf_openpose_2d['positions_2d'].item()['S1']['Directions 1']
kp2 = golf_detectron_2d['positions_2d'].item()['S1']['Directions 1']

def show_comparison_of_2d():
    for i in range(17):
        kp1_x = kp1[0][:, i, 0]
        kp1_y = kp1[0][:, i, 1]
        kp1_conf = kp1[0][:, i, 2]

        kp2_x = kp2[0][:, i, 0]
        kp2_y = kp2[0][:, i, 1]
        kp2_conf = kp2[0][:, i, 2]
        plt.plot(range(len(kp1_x)), kp1_x, label='openpose_x')
        plt.plot(range(len(kp2_x)), kp2_x, label='detectron_x')

        plt.legend()
        plt.show()

        plt.plot(range(len(kp1_y)), kp1_y, label='openpose_y')
        plt.plot(range(len(kp2_y)), kp2_y, label='detectron_y')

        plt.legend()
        plt.show()

        plt.plot(range(len(kp1_conf)), kp1_conf, label='openpose_conf')
        plt.plot(range(len(kp2_conf)), kp2_conf, label='detectron_conf')

        plt.legend()
        plt.show()

show_comparison_of_2d()

def show_comparison_of_3d():
    pred1 = golf_openpose_3d['arr_0']
    pred2 = golf_detectron_3d['arr_0']
    for i in range(17):
        joint1_x = pred1[:, i, 0]
        joint1_y = pred1[:, i, 1]
        joint1_z = pred1[:, i, 2]
        joint2_x = pred2[:, i, 0]
        joint2_y = pred2[:, i, 1]
        joint2_z = pred2[:, i, 2]
        plt.plot(range(len(joint1_x)), joint1_x, label='openpose_x')        
        plt.plot(range(len(joint2_x)), joint2_x, label='detectron_x')
        
        plt.legend()
        plt.show()
        
        plt.plot(range(len(joint1_y)), joint1_y, label='openpose_y')
        plt.plot(range(len(joint2_y)), joint2_y, label='detectron_y')
        
        plt.legend()
        plt.show()

        plt.plot(range(len(joint1_z)), joint1_z, label='openpose_z')
        plt.plot(range(len(joint2_z)), joint2_z, label='detectron_z')

        plt.legend()
        plt.show()
show_comparison_of_3d()
