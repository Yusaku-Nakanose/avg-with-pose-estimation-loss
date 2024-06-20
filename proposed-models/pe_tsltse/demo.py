from re import S
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

import time

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')
import os

#number = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]
#number = ["01_0", "01_1", "01_2", "02_0", "02_1", "02_2", "03_0", "03_1", "03_2", "04_0", "04_1", "04_2", "05_0", "05_1", "05_2", "06_0", "06_1", "06_2", "07_0", "07_1", "07_2", "08_0", "08_1", "08_2", "09_0", "09_1", "09_2", "10_0", "10_1", "10_2", "11_0", "11_1", "11_2", "12_0", "12_1", "12_2", "13_0", "13_1", "13_2", ]
#number = ["01_0", "02_0", "03_0", "04_0", "04_1", "04_2", "05_0", "05_1", "05_2", "06_0", "06_1", "06_2", "07_0", "07_1", "07_2", "08_0", "08_1", "08_2", "09_0", "09_1", "09_2", "10_0", "10_1", "10_2", "12_0", "12_1", "12_2", "13_0", "13_1", "13_2"]
#number = ["11_0", "11_1", "11_2"]
number = ["02_1", "02_2"]

for j in range(len(number)):
    # 画像が含まれるディレクトリのパス
    image_directory = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/' + number[j] + '/test_latest/images'

    # 処理結果を保存するディレクトリのパス
    output_directory1 = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/' + number[j] + '/test_OP/'
    output_directory2 = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/' + number[j] + '/test_OP_white/'

    # output_directoryが存在しない場合は作成
    if not os.path.exists(output_directory1):
        os.makedirs(output_directory1)
    
    if not os.path.exists(output_directory2):
        os.makedirs(output_directory2)

    # 画像ディレクトリ内のすべてのファイルを取得
    image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

    # 画像ディレクトリ内のすべてのファイルを取得し、ソート

    sorted_image_files = sorted(image_files, key=lambda x: (int((x.split('_')[0][-2:])), int(x.split('_')[1])))
    #sorted_image_files = sorted(image_files, key=lambda x: (int((x.split('_')[1][-2:])), int(x.split('_')[2]))) #double_bassの場合

    # 合計時間を保持する変数
    total_body_estimation_time = 0.0
    total_hand_estimation_time = 0.0

    # 画像ごとに処理を行う
    for image_file in sorted_image_files:
        
        # ファイル名に "image2" が含まれていない場合はスキップ
        if "image2" not in image_file:
            continue
        
        # 各画像のパス
        test_image = os.path.join(image_directory, image_file)

        # 以降、先程の処理をそのまま適用
        oriImg = cv2.imread(test_image)  # B,G,R order
        
        # body_estimationの開始時間
        #start_time = time.time()

        candidate, subset = body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)


        # 新たに背景画像を生成（ここでは白地の画像）
        background = np.ones_like(oriImg) * 255  # 255で白い画像を生成

        canvas1 = util.draw_bodypose(canvas, candidate, subset)
        canvas2 = util.draw_bodypose(background, candidate, subset)

        #body_estimation_time = time.time() - start_time
        #total_body_estimation_time += body_estimation_time

        """# hand_estimationの開始時間
        start_time = time.time()
        hands_list = util.handDetect(candidate, subset, oriImg)

        if subset.shape == (0, 20):
            subset = np.full(20, -1)

        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, -1, peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, -1, peaks[:, 1]+y)
            all_hand_peaks.append(peaks)

        canvas = util.draw_handpose(canvas, all_hand_peaks)

        hand_estimation_time = time.time() - start_time
        total_hand_estimation_time += hand_estimation_time"""

        """result = np.full(60, -1)
        if np.array(hands_list).shape[0] == 2:
            for i in range(18):
                if not subset[0][i] == -1:
                    result[i] = 1
                else:
                    result[i] = 0
            for i in range(21):
                if not all_hand_peaks[0][i][0] == -1:
                    result[i+18] = 1
                else:
                    result[i+18] = 0
            for i in range(21):
                if not all_hand_peaks[1][i][0] == -1:
                    result[i+39] = 1
                else:
                    result[i+39] = 0

        elif np.array(hands_list).shape[0] == 1:
            if hands_list[0][3] == True:
                for i in range(18):
                    if not subset[0][i] == -1:
                        result[i] = 1
                    else:
                        result[i] = 0
                for i in range(21):
                    if not all_hand_peaks[0][i][0] == -1:
                        result[i+18] = 1
                    else:
                        result[i+18] = 0
                for i in range(21):
                    result[i+39] = 0

            else:
                for i in range(18):
                    if not subset[0][i]== -1:
                        result[i] = 1
                    else:
                        result[i] = 0
                for i in range(21):
                    result[i+18] = 0
                for i in range(21):
                    if not all_hand_peaks[0][i][0] == -1:
                        result[i+39] = 1
                    else:
                        result[i+39] = 0
        
        elif np.array(hands_list).shape[0] == 0:
            for i in range(18):
                if not subset[0][i]== -1:
                    result[i] = 1
                else:
                    result[i] = 0
            for i in range(21):
                result[i+18] = 0
            for i in range(21):
                result[i+39] = 0

        print(f"{image_file}:[{result}")"""
        cv2.imwrite(os.path.join(output_directory1, "OP_" + image_file), canvas1)
        cv2.imwrite(os.path.join(output_directory2, "OP_white_" + image_file), canvas2)

        # 画像ファイル名と結果を記録するテキストファイルのパス
        #output_file_path = os.path.join(output_directory, number[j] + "result.txt")

        # 結果を文字列に変換
        #result_str = " ".join(map(str, result))

        # ファイルを書き込みモードで開く
        """with open(output_file_path, "a") as f:  # "a"は追記モードを示します
            # 画像ファイル名と結果をファイルに書き込む
            f.write(f"{image_file}:[{result_str}]\n")"""

    # 合計時間を表示
    """with open(output_file_path, "a") as f:
        f.write(f"Total Body Estimation Time: {total_body_estimation_time} seconds")
        f.write(f"Total Hand Estimation Time: {total_hand_estimation_time} seconds")
    print(f"Total Body Estimation Time: {total_body_estimation_time} seconds")
    print(f"Total Hand Estimation Time: {total_hand_estimation_time} seconds")"""