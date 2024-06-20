import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')
INPUT_FILE_NAME = 'epoch001_real_image.png'
test_image = 'images/' + INPUT_FILE_NAME
oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)
canvas = copy.deepcopy(oriImg)
canvas = util.draw_bodypose(canvas, candidate, subset)
# detect hand
hands_list = util.handDetect(candidate, subset, oriImg)
body_part_names = [
        "Nose     ", "Neck     ", "R-Shoulder", "R-Elbow   ", "R-Wrist   ", "L-Shoulder",
        "L-Elbow   ", "L-Wrist   ", "R-Hip     ", "R-Knee    ", "R-Ankle  ", "L-Hip    ", 
        "L-Knee   ", "L-Ankle  ", "R-Eye    ", "L-Eye    ", "R-Ear    ", "L-Ear    "
    ]

"""j = 0
for i in range(18):
    if subset[0][i] == -1:
        print(f"{i} {body_part_names[i]}: x=null, y=null")
    else:
        print(f"{i} {body_part_names[i]}: x={candidate[j][0]}, y={candidate[j][1]}")
        j += 1"""

all_hand_peaks = []
for x, y, w, is_left in hands_list:
    # (x,y):手の大雑把な位置, w:手の向きに対しての横幅, is_left:左手かどうかのブール値(True→左手, False→右手or存在しない)
    # 左手に関してと，右手に関しての二つの検出(検出できない場合は，x=y=w=0, is_left=False)
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if is_left:
        # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
        # plt.show()
    peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])# 手がある位置を指定して検出
    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    # else:
    #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
    #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
    #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    #     print(peaks)
    all_hand_peaks.append(peaks)

hand_part_names = [
    "HandBase", "ThumbBase", "ThumbJoint2", "ThumbJoint1", "ThumbTip", 
    "LittleBase", "LittleJoint2", "LittleJoint1", "LittleTip",
    "RingBase", "RingJoint2", "RingJoint1", "RingTip",
    "MiddleBase", "MiddleJoint2", "MiddleJoint1", "MiddleTip",
    "LittleBase", "LittleJoint2", "LittleJoint1", "LittleTip",
    "IndexBase", "IndexJoint2", "IndexJoint1", "IndexTip"
]

canvas = util.draw_handpose(canvas, all_hand_peaks)

cv2.imwrite("result/OP_" + INPUT_FILE_NAME, canvas)