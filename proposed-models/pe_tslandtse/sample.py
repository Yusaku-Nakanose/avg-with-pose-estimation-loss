import os
import cv2
import copy
from src import util
from src.body import Body
from src.hand import Hand
import torch
import numpy as np

INPUT_FILE_NAME = "ski.jpg"
INPUT_FILE_NAME = "/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/img/img_b/train/bassoon00_500.jpg"
INPUT_FILE_NAME = "hand.jpg"
if __name__ == "__main__":
    body_estimation = Body('model/body_pose_model.pth')
    hand_estimation = Hand('model/hand_pose_model.pth')

    target_image_path = 'images/' + INPUT_FILE_NAME
    oriImg = cv2.imread(INPUT_FILE_NAME)  # 画像の読み込み
    if oriImg is not None:
        print("画像が正しく読み込まれました。")
    else:
        print("画像の読み込みに失敗しました。")
    #print(oriImg)
    #print(type(oriImg))
    #print(oriImg.shape)
    
    
    
    #oriImg = torch.from_numpy(oriImg.astype(np.float32)).clone()
    #print(oriImg)
    #print(type(oriImg))
    #print(oriImg.shape)

    candidate, subset = body_estimation(oriImg)  # サブセット情報を無視

    # body_keypointsには各候補点のx, y座標が含まれています
    #for i, (x, y) in enumerate(body_keypoints):
    #    print(f"候補点 {i}: x={x}, y={y}")

    """body_part_names = [
        "Nose     ", "Neck     ", "RShoulder", "RElbow   ", "RWrist   ", "LShoulder",
        "LElbow   ", "LWrist   ", "RHip     ", "RKnee    ", "RAnkle  ", "LHip    ", 
        "LKnee   ", "LAnkle  ", "REye    ", "LEye    ", "REar    ", "LEar    "
    ]

    j = 0
    for i in range(18):
        if subset[0][i] == -1:
            print(f"{i} {body_part_names[i]}: x=null, y=null")
        else:
            print(f"{i} {body_part_names[i]}: x={candidate[j][0]}, y={candidate[j][1]}")
            j += 1"""

    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    basename_name = os.path.splitext(os.path.basename(target_image_path))[0]

    result_image_path = "result/pose_" + basename_name + ".png"
    cv2.imwrite(result_image_path, canvas)
