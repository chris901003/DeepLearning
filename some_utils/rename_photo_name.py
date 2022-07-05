import os


# 照片檔案路徑
data_path = '/Volumes/Pytorch/traffic_red_line'
files_name = os.listdir(data_path)
# 會先依據原始圖像名稱進行排序
files_name.sort()
total = 0

for index, file_name in enumerate(files_name):
    # 過濾一些不需要的檔案(Mac解壓縮時可能會夾帶一些不需要的檔案)
    if file_name[0] == '.':
        continue
    name = file_name.split('.')
    old_name = os.path.join(data_path, file_name)
    # 這裡的名稱是依照當前index
    new_name = os.path.join(data_path, str(index) + '.' + name[1].lower())
    # 更新檔案名稱
    os.rename(old_name, new_name)
    total += 1

print(f"Total change {total} files name.")
