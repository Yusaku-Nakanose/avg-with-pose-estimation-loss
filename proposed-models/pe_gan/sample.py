import os
import cv2
import copy
from src import util
from src.body import Body
from src.hand import Hand
import torch
import numpy as np

INPUT_FILE_NAME = "pexels-august-de-richelieu-4427816.jpg"
if __name__ == "__main__":
    body_estimation = Body('model/body_pose_model.pth')
    hand_estimation = Hand('model/hand_pose_model.pth')

    target_image_path = 'images/' + INPUT_FILE_NAME
    oriImg = cv2.imread(target_image_path)  # 画像の読み込み
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
        "Nose     ",  "Neck     ",  "RShoulder",  "RElbow   ",  "RWrist   ",  "LShoulder",
        "LElbow   ",  "LWrist   ",  "RHip     ",  "RKnee    ",  "RAnkle  ",   "LHip    ", 
        "LKnee   ",   "LAnkle  ",   "REye    ",   "LEye    ",   "REar    ",   "LEar    "
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
    #print(f"subset\n{subset}")
    #print(f"candidate\n{candidate}")
    result_image_path = "result/pose_" + basename_name + ".png"
    cv2.imwrite(result_image_path, canvas)


"""candidate(x座標, y座標, score, id) (epoch001_real_image.pngの場合)
    [[ 75.          18.           0.93580699   0.        ]
    [ 64.          35.           0.87293214   1.        ]
    [ 54.          32.           0.80147469   2.        ]
    [ 41.          66.           0.8485564    3.        ]
    [ 50.          92.           0.81267732   4.        ]
    [ 73.          37.           0.83158582   5.        ]
    [ 73.          62.           0.60644388   6.        ]
    [ 74.          75.           0.68157786   7.        ]
    [ 53.          89.           0.5989483    8.        ]
    [ 50.         127.           0.40235627   9.        ]
    [ 67.          90.           0.60216302  10.        ]
    [ 64.         127.           0.28582168  11.        ]
    [ 71.          13.           0.97364187  12.        ]
    [ 78.          14.           0.71210843  13.        ]
    [ 62.          12.           0.94865572  14.        ]]"""
