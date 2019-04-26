import json
import sys
import glob
import os
import numpy as np


def scale_down_prob_score(keypoint):
    keypoint[:, -1] *= .03
    # print(keypoint)
    return keypoint


def edit_to_coco(keypoint):
    # nose: 0, 0
    # left_eye: 1, 15
    # right_eye: 2, 14
    # left_ear: 3, 16
    # right_ear: 4, 17
    # left_shoulder: 5, 5
    # right_shoulder: 6, 2
    # left_elbow: 7, 6
    # right_elbow: 8, 3
    # left_wrist: 9, 7
    # right_wrist: 10, 4
    # left_hip: 11, 11
    # right_hip: 12, 8
    # left_knee: 13, 12
    # right_knee: 14, 9
    # left_ankle: 15, 13
    # right_ankle: 16, 10
    openpose_to_coco = [0, 15, 14, 16, 17, 5,
                        2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    # print(keypoint)
    new_keypoint = np.zeros((17, 3))
    for i in range(17):
        new_keypoint[i] = keypoint[openpose_to_coco[i]]
    # print("after", new_keypoint)
    return new_keypoint


def save_to_npz(name, output_dir):
    files = glob.glob(name + '*.json')

    glob_keypoints = []
    for json_file in files:
        with open(json_file) as keypoint:
            data = json.load(keypoint)
            pose_keypoints_2d = np.asarray(
                data['people'][0]['pose_keypoints_2d'])
            pose_keypoints_2d = np.reshape(pose_keypoints_2d, (18, 3))
            pose_keypoints_2d = edit_to_coco(pose_keypoints_2d)
            pose_keypoints_2d = scale_down_prob_score(pose_keypoints_2d)
            glob_keypoints.append(pose_keypoints_2d)

    print(np.asarray([glob_keypoints]))
    print(np.asarray([glob_keypoints]).shape)

    dictionary_keypoints = {
        'S1': {'Directions 1': np.asarray([glob_keypoints])}}
    metadata = {
        'layout_name': 'coco',
        'num_joints': 17,
        'keypoints_symmetry': [
            [1, 3, 5, 7, 9, 11, 13, 15],
            [2, 4, 6, 8, 10, 12, 14, 16],
        ]
    }
    np.savez(os.path.join(output_dir, "data_2d_detections_" + name +
                          ".npz"), metadata=metadata, positions_2d=dictionary_keypoints)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python3 save_to_npz.py keypoint_file_prefix output_dir")
    else:
        scale_down_only(sys.argv[1], sys.argv[2])
