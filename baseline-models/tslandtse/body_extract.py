import os
import cv2
import copy
import numpy as np
import glob
from src import model
from src import util
from src.body import Body

# ディレクトリ内の画像ファイル名を取得
##directory_path = '/mnt/feoh-public/sig4share/students/M1/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/05_op_3/test_latest/images/*fake_image2*.png'
directory_path = 'gif_clarinet01_128/*.png'

# OpenPose モデルの初期化
body_estimation = Body('model/body_pose_model.pth')

# 各画像に対して処理を実行
image_files = glob.glob(directory_path)
print(f"Number of image files found: {len(image_files)}")

for INPUT_FILE_NAME in image_files:
    test_image = INPUT_FILE_NAME # パスは既に取得されているので直接使用
    oriImg = cv2.imread(test_image) # B,G,R order
    
    if oriImg is None:
        print(f"Failed to read image: {INPUT_FILE_NAME}")
        continue
    
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    
    # 白地に姿勢のみの画像を作成
    white_canvas = np.ones((oriImg.shape[0], oriImg.shape[1], 3), dtype=np.uint8) * 255
    white_canvas = util.draw_bodypose(white_canvas, candidate, subset)
    
    # 出力先のディレクトリを指定して保存
    output_directory = 'gif_clarinet01_128/images_pose/'
    output_directory_white = 'gif_clarinet01_128/images_pose_white/'
    
    # 出力ディレクトリが存在するか確認し、存在しない場合は作成
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(output_directory_white, exist_ok=True)
    
    # 入力ファイル名からパスを抽出し、それを元に出力ファイルのパスを生成
    input_filename = os.path.basename(INPUT_FILE_NAME)
    output_file_path = os.path.join(output_directory, 'OP_' + input_filename)
    output_file_path_white = os.path.join(output_directory_white, 'OP_white_' + input_filename)
    
    cv2.imwrite(output_file_path, canvas)
    cv2.imwrite(output_file_path_white, white_canvas)
    
    print(f"Processed image: {INPUT_FILE_NAME}")
    print(f"Saved images: {output_file_path}, {output_file_path_white}")