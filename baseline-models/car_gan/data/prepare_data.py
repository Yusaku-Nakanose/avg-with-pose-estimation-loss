import os
import re
import numpy as np

IMG_EXTENSION = ['.jpg', 'png']

BASSOON = '1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
CELLO = '0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
CLARINET = '0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0'
DOUBLE_BASS = '0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0'
FLUTE = '0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0'
HORN = '0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0'
OBOE = '0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0'
SAX = '0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0'
TROMBONE = '0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0'
TRUMPET = '0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0'
TUBA = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0'
VIOLA = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0'
VIOLIN = '0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1'

re_digits = re.compile(r'(\d+)')


def embedded_numbers(s):
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces


def sort_string(lst):
    return sorted(lst, key=embedded_numbers)


def add_label(path):
    if path.upper().__contains__('BASSOON'):
        return BASSOON
    elif path.upper().__contains__('CELLO'):
        return CELLO
    elif path.upper().__contains__('CLARINET'):
        return CLARINET
    elif path.upper().__contains__('DOUBLE_BASS'):
        return DOUBLE_BASS
    elif path.upper().__contains__('FLUTE'):
        return FLUTE
    elif path.upper().__contains__('HORN'):
        return HORN
    elif path.upper().__contains__('OBOE'):
        return OBOE
    elif path.upper().__contains__('SAX'):
        return SAX
    elif path.upper().__contains__('TROMBONE'):
        return TROMBONE
    elif path.upper().__contains__('TRUMPET'):
        return TRUMPET
    elif path.upper().__contains__('TUBA'):
        return TUBA
    elif path.upper().__contains__('VIOLA'):
        return VIOLA
    elif path.upper().__contains__('VIOLIN'):
        return VIOLIN
    else:
        print("LABEL ERROR")
        return False


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSION)


def make_dataset(dir, max_dataset_size=float("inf")):
    assert os.path.isfile(dir), '%s is not a valid txt file' % dir
    audio = []
    label = []
    image = []

    with open(dir, 'r') as f:
        for line in f.readlines():
            line_list = line.split('||')
            audio.append(line_list[0])
            label.append(np.fromstring(line_list[1], dtype=float, sep=','))
            image.append(line_list[2].split('\n')[0])
    return {'audio': audio[:min(max_dataset_size, len(audio))],
            'label': label[:min(max_dataset_size, len(label))],
            'image': image[:min(max_dataset_size, len(image))]}


def prepare_data(dir):
    global i, j
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    filepaths = []
    num = 1000
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sort_string(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                #../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_con/train/violin/violin01_500_con.png
                #../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img/train/violin/violin02_500.jpg

                #########concate使う時に数が合わなくなるから修正(6/20)
                if "spec" in path:
                    if "train" in path:
                        #print("koko")
                        if "violin02" in path:
                            path_name_a = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_v_02/train/violin/violin02_"
                            path_name_b =".png"
                            filepaths.append(path_name_a + str(num) + path_name_b)
                            num = num + 100
                            """ path_img = path.split("spec")[0] + "img/train/" + path.rsplit("/",2)[1] + "/" + path.split("/")[-1]
                            print(path_img)
                            print(path.split("/")[-1])
                            print(a)
                            if os.path.exists(path_img.replace("_con","").replace("png","jpg")):
                                filepaths.append(path) """
                    elif "test" in path:
                        if "violin06" in path:
                            path_name_a = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_v_02/test/violin/violin06_"
                            path_name_b =".png"
                            filepaths.append(path_name_a + str(num) + path_name_b)
                            num = num + 100
                            """ path_img = path.split("spec")[0] + "img/test/" + path.rsplit("/",2)[1] + "/" + path.split("/")[-1]
                            if os.path.exists(path_img.replace("_con","").replace("png","jpg")):
                                filepaths.append(path) """
                    #print(path_img.replace("_con",""))
                    #print(a)
                    #print(path_img.replace("png","jpg"))
                    #print(os.path.exists(path_img.replace("png","jpg")))
                    """ if os.path.exists(path_img.replace("_con","").replace("png","jpg")):
                        filepaths.append(path) """
                    
                if "img" in path:
                    if "train" in path:
                        if "violin02" in path:
                            """ path_img = path.split("img")[0] + "spec_con/train/" + path.rsplit("/",2)[1] + "/" + path.split("/")[-1] """
                            path_name_a = "../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img/train/violin/violin02_"
                            path_name_b =".jpg"
                            filepaths.append(path_name_a + str(num+500) + path_name_b)
                            num = num + 100
                        if num > 130400:
                            break
                            
                    elif "test" in path:
                        if "violin06" in path:
                            #path_img = path.split("img")[0] + "spec_con/test/" + path.rsplit("/",2)[1] + "/" + path.split("/")[-1]
                            path_name_a = "../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img/test/violin/violin06_"
                            path_name_b =".jpg"
                            filepaths.append(path_name_a + str(num+500) + path_name_b)
                            num = num + 100
                        if num > 29500:
                            break
                    
                    #if os.path.exists(path_img.replace(".jpg","_con.png")):
                    #    filepaths.append(path)
                """ if num > 30000:
                    break """    
                #########################################################
                #filepaths.append(path)
    return filepaths

def prepare_label(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    filepaths = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sort_string(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                filepaths.append(add_label(path))
    return filepaths

def prepare_all(audio, image, trainOrtesr):
    audios = prepare_data(audio)
    images = prepare_data(image)
    labels = prepare_label(audio)
    size = len(audios)
    
    with open('/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/' + trainOrtesr + '_zurashi_v_02_data.txt', 'w+') as f:
        count = 0
        for laudio, limage, llabel in zip(audios, images, labels):
            f.write(laudio + '||')
            f.write(llabel + '||')
            f.write(limage)
            count += 1
            if count < size:
                f.write('\n')


if __name__ == '__main__':
    """ train_audio = '../sample_data/spectram/train/'
    train_image = '../sample_data/img_256/train/' """

    """ test_audio = '../sample_data/spectram/test/'
    test_image = '../sample_data/img_256/test/' """


    ################violin　オンリー###############
    train_audio = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec/train"
    train_image = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img/train"
    #prepare_all(train_audio, train_image, 'train')

    test_audio = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec/test"
    test_image = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img/test"
    prepare_all(test_audio, test_image, 'test')

     
    ################violin_con　オンリー###############
    """ train_audio = "/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_con/train"
    train_image = "/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img/train"
    prepare_all(train_audio, train_image, 'train') """

    """ test_audio = "/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_con/test"
    test_image = "/mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img/test"
    prepare_all(test_audio, test_image, 'test') """

    ################楽器ミックス###############
    """ train_audio = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/spec_m/train"
    train_image = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/car_gan/img_m/train"
    prepare_all(train_audio, train_image, 'train')


    test_audio = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/spec/validation"
    test_image = "../../../../../mnt/othel-public/sig4share/students/M2/nakagawa/m1/data_URMP/Sub-URMP/img/validation"
    prepare_all(test_audio, test_image, 'test') """
