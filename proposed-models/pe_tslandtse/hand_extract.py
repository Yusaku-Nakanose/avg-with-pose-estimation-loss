import os

import cv2
import copy
import numpy as np
import glob

from src import model
from src import util
from src.body import Body
from src.hand import Hand

# ディレクトリ内の画像ファイル名を取得
directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/img/img_b/test/*'

# OpenPose モデルの初期化
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

num = 0
i = 0

# 各画像に対して処理を実行
for INPUT_FILE_NAME in glob.glob(directory_path):
    i += 1
    test_image = INPUT_FILE_NAME  # パスは既に取得されているので直接使用
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    print(f"candidate:{candidate}")
    print(f"subset:{subset}")
    keypoints = [[-1, -1]] * 18


    # bの-1に対応するデータをaに追加
    for i in range(18):
        if subset[0, i] == -1:
            candidate = np.insert(candidate, i, [-1, -1, -1, -1], axis=0)

    keypoints = candidate[:, :2]

    print(f"keypoints:{keypoints}")
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0] == 0, -1, peaks[:, 0] + x)
        peaks[:, 1] = np.where(peaks[:, 1] == 0, -1, peaks[:, 1] + y)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)
    all_hand_peaks = np.array(all_hand_peaks)
    all_hand_peaks.reshape(42, 2)
    print(all_hand_peaks)
    #all_hand_peaks_flat = all_hand_peaks.flatten()

    # 正しく配列の比較を行うために、`[-1, -1]` を `np.array([-1, -1])` に変更
    for a in range(2):
        for b in range(21):
            if np.array_equal(all_hand_peaks[a][b], np.array([-1, -1])):
                num += 1

    # 出力先のディレクトリを指定して保存
    output_directory = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/OpenPose_result/bassoon/test/'
    
    # 入力ファイル名からパスを抽出し、それを元に出力ファイルのパスを生成
    input_filename = os.path.basename(INPUT_FILE_NAME)
    output_file_path = os.path.join(output_directory, 'OP_' + input_filename)
    cv2.imwrite(output_file_path, canvas)

print(num)
print(i)
