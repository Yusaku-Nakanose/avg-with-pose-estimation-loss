from PIL import Image
import os
import glob

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
""" date = "0909_3_2"
create_gif(in_dir="/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/test_latest/images/", out_filename="/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/" + date + "/gif_trom_2_" + date + "_10000to20000_loop1_quan.gif") """

""" instru = "o_0908_1"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "o_0908_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "tp_0907_1"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "tp_0907_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "c_0906_1_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "c_0906_2_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "cl_0905_1"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "cl_0905_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "h_0904_1"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "h_0904_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "b_0903_1"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "b_0903_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "f_0902_1"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "f_0902_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "s_0901_1"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "s_0901_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR) """

""" instru = "tb_1909_3"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "vi_1910_3"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR) """

""" instru = "d_0911_1"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "d_0911_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "tu_0912_1"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "tu_0912_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "vl_0913_1"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "vl_0913_2"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_10000to20000_loop1_quan.gif"
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)
 """


########

#GTのgif作る用
""" instru = "bassoon01"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+ instru +"_128/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_128_gt_10000to20000_loop1_quan.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR) """

""" instru = "cello01"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_gt_10000to20000_loop1_quan.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "clarinet01"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_gt_10000to20000_loop1_quan.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "flute01"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_gt_10000to20000_loop1_quan.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "horn04"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_gt_10000to20000_loop1_quan.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "sax01"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_gt_10000to20000_loop1_quan.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

instru = "trumpet01"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_gt_10000to20000_loop1_quan.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR) 

instru = "oboe04_128"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+ instru +"/"
OUT_DIR = "/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_gt_10000to20000_loop1_quan.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)"""

instru = "flute01_128"
INPUT_DIR = "/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_f_4902_2"
#INPUT_DIR = "/mnt/feoh-public\sig4share\students\卒業生\2022\nakagawa\m1\data_URMP\Sub-URMP\car_gan\results\gif_f_4902_2"
OUT_DIR = "/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/gif_" + instru +"_gt_10000to20000_loop1_quan_nakagawa.gif"
#print(OUT_DIR)
create_gt_gif(in_dir=INPUT_DIR, out_filename=OUT_DIR)

#create_gt_gif(in_dir="/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_o_00/test/", out_filename="/mnt/feoh-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_oboe04_gt_10000to20000_loop1_quan.gif")
