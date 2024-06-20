#GT用
#pe_tsltseの方に生成動画用のコードがある

from PIL import Image
import os
import argparse
import glob

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--date', type=str, default='0000_0')
parser.add_argument('-i','--inst', type=str, default='xx')
opt = parser.parse_args()

# GIFアニメーションを作成
def create_gif(in_dir, out_filename):
    #path_list = sorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
    path_list = sorted(glob.glob(os.path.join(*[in_dir, '*fake_image2_1.png'])))
    #print(len(path_list))
    imgs = []                                                   # 画像をappendするための空配列を定義
    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in range(len(path_list)):
        img = Image.open(path_list[i]).quantize()               # 画像ファイルを1つずつ開く
        img = img.resize((128,128))
        imgs.append(img)                                        # 画像をappendで配列に格納していく
        if i > 98:
            break
    
    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=100, loop=1)

def create_gt_gif(in_dir, out_filename):
    #path_list = sorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
    path_list = sorted(glob.glob(os.path.join(*[in_dir, '*'])))
    imgs = []                                                   # 画像をappendするための空配列を定義

    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in range(len(path_list)):
        img = Image.open(path_list[i]).quantize()               # 画像ファイルを1つずつ開く
        img = img.resize((128,128))
        imgs.append(img)                                        # 画像をappendで配列に格納していく
        if i > 98:
            break
    
    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=100, loop=1, quality = 95)

# GIFアニメーションを作成する関数を実行する
#date = "2905_1"
#create_gif(in_dir="/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/test_latest/images/", out_filename="/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/gif_cl_" + date + "_10000to20000_loop1_quan.gif")


# GIFアニメーションを作成する関数を実行する
date = opt.date
create_gif(in_dir="/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/GT_test_gif/test_OP/" + date, out_filename="/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/GT_test_gif/gif_"+opt.inst+"_" + date + "_OP_10000to20000_loop1_quan.gif")
create_gif(in_dir="/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/GT_test_gif/test_OP_white/" + date, out_filename="/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/GT_test_gif/gif_"+opt.inst+"_" + date + "_OP_white_10000to20000_loop1_quan.gif")


#GTのgif作る用
""" create_gt_gif(in_dir="/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_trombone02/", out_filename="/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_trombone02_gt_10000to20000_loop1_quan.gif") """


