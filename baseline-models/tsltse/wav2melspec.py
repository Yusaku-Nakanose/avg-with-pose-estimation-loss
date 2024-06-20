import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob 
import os
import re

#OTHEL_PATH = "../../../../mnt/othel-public/sig4share/students/M1/nakagawa/m1/data_URMP/Sub-URMP/chunk/"
FEOH_PATH = "/mnt/feoh-public/sig4share/students/B4/nakanose/b4/data_URMP/Sub_URMP/chunk/"
wav_file = glob.glob(FEOH_PATH+"validation/*/*")
print(len(wav_file))
for i in range(len(wav_file)):
    print(wav_file[i])
    wave, fs = librosa.load(wav_file[i])
    n_mels = 128
    fmax = 8000
    mel = librosa.feature.melspectrogram(y=wave, sr=fs, n_fft=2048, hop_length=512)
    #mel_dB = librosa.amplitude_to_db(mel, ref=np.max) #dBに変換 うまくいかない？
    plt.rcParams["figure.figsize"] = (2.86, 2.86) #256x256
    mel_dB = 20 * np.log10(mel) #相対的じゃないやつ
    librosa.display.specshow(mel_dB, cmap = 'jet')
    plt.axis('off')
    plt.tight_layout()

    OUTPUT_PATH = wav_file[i].replace("chunk","spec").replace("wav","png")
    plt.savefig(OUTPUT_PATH, bbox_inches='tight',pad_inches = 0)
    plt.clf()
    plt.close()


############1つだけ試すとき####################
""" wave, fs = librosa.load("./violin06_2400.wav")
n_mels = 128
fmax = 8000
mel = librosa.feature.melspectrogram(y=wave, sr=fs, n_fft=2048, hop_length=512)
#mel_dB = librosa.amplitude_to_db(mel, ref=np.max) #dBに変換
plt.rcParams["figure.figsize"] = (2.86, 2.86) #256x256
mel_dB = 20 * np.log10(mel)
librosa.display.specshow(mel_dB, cmap = 'jet')
plt.axis('off')
plt.tight_layout()

#OUTPUT_PATH = wav_file[i].replace("chunk","spec").replace("wav","png")
plt.savefig("./violin06_2400.png", bbox_inches='tight',pad_inches = 0) """