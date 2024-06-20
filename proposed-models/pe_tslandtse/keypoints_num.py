import os
import cv2
import glob
from src.body import Body
import numpy as np

# OpenPose モデルの初期化
body_estimation = Body('model/body_pose_model.pth')

# 処理する番号のリスト
#numbers = ["01_1", "02_1", "03_1", "04_1", "05_1", "06_1", "07_1", "08_1", "09_1", "10_1", "11_1", "12_1", "13_1"]
#numbers = ["01_2", "02_2", "03_2", "04_2", "05_2", "06_2", "07_2", "08_2", "09_2", "10_2", "11_2", "12_2", "13_2"]
#numbers = ["01_op_1", "02_op_1", "03_op_1", "04_op_1", "05_op_1", "06_op_1", "07_op_1", "08_op_1", "09_op_1", "10_op_1", "11_op_1", "12_op_1", "13_op_1"]
numbers = ["01_op_2", "02_op_2", "03_op_2", "04_op_2", "05_op_2", "06_op_2", "07_op_2", "08_op_2", "09_op_2", "10_op_2", "11_op_2", "12_op_2", "13_op_2"]
#numbers = ["gif_bassoon01_128", "gif_cello01_128", "gif_clarinet01_128", "gif_double_bass01_128", "gif_flute01_128", "gif_horn04_128", "gif_oboe04_128", "gif_sax01_128", "gif_trombone02_128", 
#           "gif_trumpet01_128", "gif_tuba04_128", "gif_viola01_128", "gif_violin06_128"]
# ベースディレクトリパス
base_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/'
#base_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_GT/'

# 出力ディレクトリのパス
#output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_analysis_tsl/'
#output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_analysis_tse/'
#output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_analysis_tsl_pe/'
output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_analysis_tse_pe/'
#output_directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/movement_analysis_gt/'

# 各楽器に対する処理
for num in numbers:
    directory_path = os.path.join(base_directory_path, num, "test_latest/images/*")
    #directory_path = os.path.join(base_directory_path, num, "*")
    image_files = sorted(glob.glob(directory_path))
    image_files = [file for file in image_files if 'fake_image2_1' in file]

    output_instrument_directory = os.path.join(output_directory_path, num)
    os.makedirs(output_instrument_directory, exist_ok=True)

    scores_file = open(os.path.join(output_instrument_directory, 'score_num.txt'), 'w')

    scores_list = []

    # 各画像に対して処理を実行
    for image_file in image_files:
        oriImg = cv2.imread(image_file)  # B,G,R order
        _, subset = body_estimation(oriImg)

        # 検出されたキーポイントの数を取得
        scores = int(subset[0, -2]) if len(subset) > 0 else 0
        scores_list.append(scores)

        # 検出されたキーポイントの数をファイルに書き込み
        scores_file.write(f"{image_file}: scores: {scores}\n")

    # キーポイントの数の分散を計算
    variance = np.var(scores_list)

    # 各カテゴリーでのキーポイントの総数と分散をファイルに書き込み
    scores_file.write(f"Average scores for {num}: {sum(scores_list)/len(image_files)}\n")
    scores_file.write(f"Variance in scores for {num}: {variance}\n")
    print(f"Average scores for {num}: {sum(scores_list)/len(image_files)}")
    print(f"Variance in scores for {num}: {variance}")

    scores_file.close()