import subprocess
import os
import io
import whatimage
import pyheif
import traceback
from PIL import Image
import tqdm


def decodeImage(bytesIo, save_photo):
    fmt = whatimage.identify_image(bytesIo)
    if fmt in ['heic', 'HEIC']:
        detail = pyheif.read_heif(bytesIo)
    pi = Image.frombytes(mode=detail.mode, size=detail.size, data=detail.data)
    pi.save(f'{save_photo}.jpg', format="jpeg")


def read_image_file_rb(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    return file_data


if __name__ == "__main__":
    path = '/Volumes/Pytorch/traffic_red_line/image2'
    save_path = '/Volumes/Pytorch/traffic_red_line/convert'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    photos_name = os.listdir(path)
    total = 0
    for photo_name in photos_name:
        if photo_name.count('.') != 1:
            continue
        if photo_name.split('.')[-1] not in ['heic', 'HEIC']:
            continue
        photo_path = os.path.join(path, photo_name)
        data = read_image_file_rb(photo_path)
        photo_name = photo_name.split('.')[0]
        save = os.path.join(save_path, photo_name)
        decodeImage(data, save)
        total += 1
    print(f'Total convert {total} photos.')
