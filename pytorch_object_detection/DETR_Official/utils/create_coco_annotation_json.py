import os
from tqdm import tqdm
import json
from PIL import Image


# 改這裡的路徑就可以了，路徑底下要有兩個資料夾，分別為images以及labels
dataset_path = './my_dataset'

# images以及labels的路徑
images_path = os.path.join(dataset_path, 'images')
labels_path = os.path.join(dataset_path, 'labels')

# info與licenses是隨便寫的如果可以不用之後可以刪除
info = {'description': 'My_dataset', 'url': 'http://', 'version': '1.0', 'year': 2022, 'contributor': 'None',
        'date_created': '2022/06/12'}
licenses = [{'url': 'http://', 'id': 1, 'name': 'None'}]

# 保存圖片相關資料
images = []
# 抓出圖片資料夾下所有圖片檔案
images_files = os.listdir(images_path)
# 排序確保每次的順序都一樣
images_files = sorted(images_files)
# 將圖片讀出來
for filename in tqdm(images_files):
    if filename.split('.')[-1] != 'jpg':
        continue
    img = Image.open(os.path.join(images_path, filename))
    w, h = img.size
    image_id = int(filename.split('.')[0])
    images.append({'license': 1, 'file_name': filename, 'coco_url': f'http://{filename}', 'height': h, 'width': w,
                   'date_captured': '2022-11-11 11:11:11', 'flickr_url': 'url:http://', 'id': image_id})

# 保存實際匡資料
annotations = []
# 抓出圖片實際匡資料夾下的檔案
annotations_files = os.listdir(labels_path)
# 排序確保每次的順序一樣
annotations_files = sorted(annotations_files)
# 每個標記匡需要一個獨立的id
annotation_id = 10000
# 將每個檔案讀出來
for annotation in tqdm(annotations_files):
    # 檔名處理
    filename = annotation.split('.')
    # 過濾不合法檔案
    if filename[0] == '' or len(filename) != 2 or filename[-1] != 'txt':
        continue
    image_id = int(filename[0])
    # 讀取檔案內容
    with open(os.path.join(labels_path, annotation), 'r') as f:
        lines = f.readlines()
        for line in lines:
            # detail = [class, center_x, center_y, w, h]
            detail = line.split(' ')
            area = float(detail[4]) * float(detail[3])
            annotations.append({
                'segmentation': [],
                'area': area,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [float(detail[1]), float(detail[2]), float(detail[3]), float(detail[4])],
                'category_id': int(detail[0]),
                'id': annotation_id
            })
            annotation_id += 1

categories_path = os.path.join(dataset_path, 'categories.names')
categories = []
with open(categories_path, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        categories.append({
            'supercategory': line,
            'id': i + 1,
            'name': line
        })

annotation_dict = {
    'info': info,
    'licenses': licenses,
    'images': images,
    'annotations': annotations,
    'categories': categories
}

with open(os.path.join(dataset_path, 'instances.json'), 'w+') as out:
    json.dumps(annotation_dict, out, indent=4)
