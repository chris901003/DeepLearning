import os
import json


json_path = '/Volumes/Pytorch/traffic_red_line/test'
json_files = os.listdir(json_path)
json_files.sort()
total = 0

for json_file in json_files:
    if json_file.count('.') != 1:
        continue
    file_path = os.path.join(json_path, json_file)
    with open(file_path) as f:
        data = json.load(f)
    file = open(file_path, 'w')
    json.dump(data, file)
    total += 1

print(f'Total change {total} files.')
