import cv2
import numpy as np
import imageio

# 光流法の初期化
def initialize_optical_flow(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(image)
    hsv[..., 1] = 255
    return gray, hsv

# 光流を計算して線だけを描画
def calculate_optical_flow(prev_gray, curr_gray, prev_points):
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None)

    lines = []
    for i, (prev, next) in enumerate(zip(prev_points, next_points)):
        x1, y1 = prev.ravel()
        x2, y2 = next.ravel()
        if status[i] == 1:  # フローが正常に計算された場合
            lines.append([(int(x1), int(y1)), (int(x2), int(y2))])

    return lines, next_points

# メインの処理
def main(input_gif_path, output_gif_path):
    # GIFの読み込み
    gif_reader = imageio.get_reader(input_gif_path)
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in gif_reader]

    # 初期化
    max_corners = 10  # 保持する特徴点の数
    min_quality_level = 0.3
    min_distance = 7

    prev_frame = frames[0]
    prev_gray, hsv = initialize_optical_flow(prev_frame)

    # 最初のフレームで特徴点を一定数検出
    prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=min_quality_level, minDistance=min_distance)
   



    # GIFの各フレームに対して光流法を適用し、線を描画
    output_lines = []
    for i in range(1, len(frames)):
        curr_frame = frames[i]
        lines, next_points = calculate_optical_flow(prev_gray, cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY), prev_points)

        # 検出された線を保存
        output_lines.append(lines)

        # 次のフレームのために更新
        prev_frame = curr_frame
        prev_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        prev_points = next_points

    # 線だけを描画して保存
    output_frames = []
    for i, frame in enumerate(frames[:-1]):
        lines = output_lines[i]
        result_frame = frame.copy()
        for line in lines:
            cv2.line(result_frame, line[0], line[1], (0, 0, 255), 2)

        output_frames.append(result_frame)

    # 結果をGIFとして保存
    imageio.mimsave(output_gif_path, output_frames, duration=0.1)

if __name__ == "__main__":
    input_gif_path = "/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/10_op_2/gif_vi_10_op_2_10000to20000_loop1_quan.gif"  # 入力GIFのファイルパスを指定
    output_gif_path = "/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/10_op_2/gif_vi_10_op_2_OF_10000to20000_loop1_quan.gif"  # 出力GIFのファイルパスを指定
    #input_gif_path = "/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/GT_test_gif/GT/gif_violin06_128_gt_10000to20000_loop1_quan.gif"
    #output_gif_path = "/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/GT_test_gif/GT/gif_violin06_OF_128_gt_10000to20000_loop1_quan.gif"
    main(input_gif_path, output_gif_path)
