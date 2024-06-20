import os

# ディレクトリのパス
directory_path = '/mnt/feoh-public/sig4share/students/b4/nakanose/b4/data_URMP/Sub-URMP/car_gan/results/hand/13_0/test_latest/images/'

# ディレクトリ内のファイルを取得
files = os.listdir(directory_path)

# 削除する対象の単語
keywords_to_remove = ['fake', 'audio']

# ディレクトリ内の各ファイルに対して処理
for file_name in files:
    # 各キーワードを含むファイルを検索
    if any(keyword in file_name for keyword in keywords_to_remove):
        # ファイルの絶対パスを取得
        file_path = os.path.join(directory_path, file_name)
        
        # ファイルを削除
        os.remove(file_path)
        