import os


data_path = '/Volumes/Pytorch/traffic_red_line'
files_name = os.listdir(data_path)
files_name.sort()
cur_file = 0
for file_name in files_name:
    if file_name[0] == '.':
        continue
    name = file_name.split('.')
    old_name = os.path.join(data_path, file_name)
    new_name = os.path.join(data_path, str(cur_file) + '.' + name[1].lower())
    os.rename(old_name, new_name)
    cur_file += 1

print(f"Total change {cur_file} files name.")
