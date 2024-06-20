# -*- coding: utf-8 -*-
from pydub import AudioSegment

# 音声ファイルの読み込み

input_OTHEL_PATH = "/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/chunk/validation/violin/"

output_OTHEL_PATH = "/mnt/feoh-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/car_gan/results/audio_connect/"

""" sound1 = AudioSegment.from_file("violin01_4000.wav", "wav")
sound2 = AudioSegment.from_file("violin01_4500.wav", "wav")
sound3 = AudioSegment.from_file("violin01_5000.wav", "wav")
sound4 = AudioSegment.from_file("violin01_5500.wav", "wav")
sound5 = AudioSegment.from_file("violin01_6000.wav", "wav")
sound6 = AudioSegment.from_file("violin01_6500.wav", "wav") """

num = 500
sound1 = AudioSegment.from_file(input_OTHEL_PATH + "violin06_" + str(10000) + ".wav", "wav")
for i in range(18):
    sound2 = AudioSegment.from_file(input_OTHEL_PATH + "violin06_" + str(10000+num*(i+1)) + ".wav", "wav")
    sound1 = sound1 + sound2.fade_in(5).fade_out(5)
    #print(10000+num*(i+1))
sound3 = AudioSegment.from_file(input_OTHEL_PATH + "violin06_" + str(19500) + ".wav", "wav")
sound1 = sound1 + sound3.fade_in(5)


""" sound1 = sound1.fade_out(5)
sound2 = sound2.fade_in(5).fade_out(5)
sound3 = sound3.fade_in(5).fade_out(5)
sound4 = sound4.fade_in(5).fade_out(5)
sound5 = sound5.fade_in(5).fade_out(5)
sound6 = sound6.fade_in(5) """

# 連結
#sound = sound1 + sound2 + sound3 + sound4 + sound5 + sound6

# 保存
sound1.export(output_OTHEL_PATH +"violin06_10000to20000_output_fi_fo_5.wav", format="wav")