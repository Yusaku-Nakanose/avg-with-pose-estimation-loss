from PIL import Image
import os
import glob

# GIFアニメーションを作成
def create_gif(in_dir, out_filename):
    #path_list = sorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
    path_list = sorted(glob.glob(os.path.join(*[in_dir, '*fake_image2.png'])))
    """ print(path_list)
    print(len(path_list)) """
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
    """ print(path_list)
    print(len(path_list)) """
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

# GIFアニメーションを作成する関数を実行する
""" date = "0909_4_2"
create_gif(in_dir="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/test_latest/images/", out_filename="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/gif_trom_2_" + date + "_10000to20000_loop1_quan.gif")

date = "0909_5_2"
create_gif(in_dir="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/test_latest/images/", out_filename="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/gif_trom_2_" + date + "_10000to20000_loop1_quan.gif")

date = "0909_6_2"
create_gif(in_dir="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/test_latest/images/", out_filename="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/gif_trom_2_" + date + "_10000to20000_loop1_quan.gif") """
date = "0910_3_2"
create_gif(in_dir="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/test_latest/images/", out_filename="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/gif_viol_2_" + date + "_10000to20000_loop1_quan.gif")

""" date = "0910_5_2"
create_gif(in_dir="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/test_latest/images/", out_filename="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/gif_viol_2_" + date + "_10000to20000_loop1_quan.gif")

date = "0910_6_2"
create_gif(in_dir="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/test_latest/images/", out_filename="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/gif_viol_2_" + date + "_10000to20000_loop1_quan.gif") """





#GTのgif作る用
""" create_gt_gif(in_dir="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_trombone02/", out_filename="/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_trombone02_gt_10000to20000_loop1_quan.gif") """


