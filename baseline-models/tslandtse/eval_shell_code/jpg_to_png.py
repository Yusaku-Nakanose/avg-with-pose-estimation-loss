from PIL import Image
import os
import glob

# GIFアニメーションを作成
def convert_jpg_to_png(in_dir, out_dir):
    path_list = sorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
    #print(path_list)
    imgs = []                                                   # 画像をappendするための空配列を定義

    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in range(len(path_list)):
        img = Image.open(path_list[i])

        img = img.resize((128,128))
        #print(path_list[i].rsplit("/")[-1].split(".")[0])
        img.save(out_dir + path_list[i].rsplit("/")[-1].split(".")[0] + "_128.png")
        #print(a)
    

""" img = Image.open("/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_trombone02/trombone02_10100.jpg").quantize()
img = img.resize((128,128))
img.save("/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_trombone02_128/trombone02_10100_128.png") """


# GIFアニメーションを作成する関数を実行する
#convert_jpg_to_png(in_dir="/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_violin06", out_dir='/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_violin06_128/')


instru="cello01"
convert_jpg_to_png(in_dir="/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+instru+"/", out_dir='/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_'+ instru +'_128/')

instru="clarinet01"
convert_jpg_to_png(in_dir="/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+instru+"/", out_dir='/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_'+ instru +'_128/')

instru="flute01"
convert_jpg_to_png(in_dir="/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+instru+"/", out_dir='/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_'+ instru +'_128/')

instru="horn04"
convert_jpg_to_png(in_dir="/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+instru+"/", out_dir='/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_'+ instru +'_128/')

instru="oboe04"
convert_jpg_to_png(in_dir="/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+instru+"/", out_dir='/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_'+ instru +'_128/')

instru="sax01"
convert_jpg_to_png(in_dir="/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+instru+"/", out_dir='/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_'+ instru +'_128/')

instru="trumpet01"
convert_jpg_to_png(in_dir="/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_"+instru+"/", out_dir='/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/gif_GT/gif_'+ instru +'_128/')