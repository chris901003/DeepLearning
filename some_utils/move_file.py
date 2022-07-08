import os
import shutil


source_path = '/Volumes/Pytorch/traffic_red_line/lableme2coco'
dis_path = '/Volumes/Pytorch/traffic_red_line/test'
source_files = os.listdir(source_path)
source_files.sort()
total = 0

for source_file in source_files:
    if source_file.count('.') != 1:
        continue
    old_path = os.path.join(source_path, source_file)
    new_path = os.path.join(dis_path, source_file)
    shutil.move(old_path, new_path)
    total += 1

print(f'Total move {total} files.')
