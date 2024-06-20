import os
import cv2
import numpy as np
import glob
import torch

from src import model
from src import util
from src.body import Body

# OpenPose モデルの初期化
body_estimation = Body('model/body_pose_model.pth')

# 楽器のリスト
#instruments = ["gif_bassoon01_128"]
numbers = ["01"]
#numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]
#numbers = ["01_1", "02_1", "03_1", "04_1", "05_1", "06_1", "07_1", "08_1", "09_1", "10_1", "11_1", "12_1", "13_1"]
#numbers = ["01_2", "02_2", "03_2", "04_2", "05_2", "06_2", "07_2", "08_2", "09_2", "10_2", "11_2", "12_2", "13_2"]
#numbers = ["01_op_1", "02_op_1", "03_op_1", "04_op_1", "05_op_1", "06_op_1", "07_op_1", "08_op_1", "09_op_1", "10_op_1", "11_op_1", "12_op_1", "13_op_1"]
#numbers = ["01_op_2", "02_op_2", "03_op_2", "04_op_2", "05_op_2", "06_op_2", "07_op_2", "08_op_2", "09_op_2", "10_op_2", "11_op_2", "12_op_2", "13_op_2"]
#numbers = ["gif_bassoon01_128", "gif_cello01_128", "gif_clarinet01_128", "gif_double_bass01_128", "gif_flute01_128", "gif_horn04_128", "gif_oboe04_128", "gif_sax01_128", "gif_trombone02_128", "gif_trumpet01_128", "gif_tuba04_128", "gif_viola01_128", "gif_violin06_128"]
# ベースディレクトリパス
#base_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_GT/'
base_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/hand/'
# 出力ディレクトリのパス

#output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_analysis_tsl/'
#output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_analysis_tse/'
#output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_analysis_tsl_pe/'
#output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_analysis_tse_pe/'
#output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/eval_sample/'
#output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_notmse/'
output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_testes/'

# MSE損失計算の関数
def criterionOP(keypoint1, keypoint2):
    total_loss = 0.0
    n = 0
    for i in range(18):
        if keypoint1[i][0] != -1 and keypoint2[i][0] != -1:
            diff_x = keypoint1[i][0] - keypoint2[i][0]
            diff_y = keypoint1[i][1] - keypoint2[i][1]
            total_loss += (diff_x**2 + diff_y**2)
            n += 1
    if n > 0:
        total_loss /= n
    return total_loss

# 各楽器に対する処理
for num in numbers:
    directory_path = os.path.join(base_directory_path, num, "test_latest/images/*")
    print(directory_path)
    all_image_files = sorted(glob.glob(directory_path))
    # 'fake_image2_1'を含むファイル名のみを選択
    image_files = [file for file in all_image_files if '_real_image' in file]
    keypoints_list = []
    movements_list = []

    output_instrument_directory = os.path.join(output_directory_path, num)
    os.makedirs(output_instrument_directory, exist_ok=True)

    keypoints_file = open(os.path.join(output_instrument_directory, 'keypoints.txt'), 'w')
    movements_file = open(os.path.join(output_instrument_directory, 'movements.txt'), 'w')

    # 各画像に対して処理を実行
    # 各画像に対して処理を実行
    for image_file in image_files:
        oriImg = cv2.imread(image_file)  # B,G,R order
        candidate, subset = body_estimation(oriImg)
        #print(candidate)
        #print(subset)
        # subsetに要素が存在するか確認
        if len(subset) > 0:
            canvas = util.draw_bodypose(oriImg, candidate, subset)

            keypoints = [[-1, -1]] * 18
            for i in range(18):
                index = int(subset[0, i])
                if index != -1:
                    keypoints[i] = candidate[index, :2].tolist()
            keypoints_list.append(keypoints)

            # keypointsをファイルに書き込み
            keypoints_file.write(f"{image_file}: {keypoints}\n")

            # 処理済み画像を保存
            output_image_file = os.path.join(output_instrument_directory, os.path.basename(image_file))
            cv2.imwrite(output_image_file, canvas)
        else:
            # subsetが空の場合は-1で埋めたキーポイントリストを作成
            keypoints = [[-1, -1]] * 18
            keypoints_list.append(keypoints)

            # keypointsをファイルに書き込み
            keypoints_file.write(f"{image_file}: {keypoints}\n")

    # 各フレーム間のMSEを計算
    total_movement = 0.0
    for i in range(len(keypoints_list) - 1):
        movement = criterionOP(keypoints_list[i], keypoints_list[i + 1])
        total_movement += movement
        movements_list.append(movement)

        # movementsをファイルに書き込み
        movements_file.write(f"Movement {i}-{i+1}: {movement}\n")

    print(f"Average Movement for {num}: {total_movement/(len(image_files)-1)}")
    keypoints_file.close()
    movements_file.close()
