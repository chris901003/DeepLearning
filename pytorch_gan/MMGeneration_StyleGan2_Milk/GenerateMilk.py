import mmcv
from mmgen.apis import init_model, sample_unconditional_model
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num_picture', default=1, type=int)
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config_file = './stylegan2_c2_ffhq_256_b4x8_800k_strawberry.py'
    checkpoint_file = './best.pth'
    save = True
    save_path = './Image'

    img_size = 256
    model = init_model(config_file, checkpoint_file, device=device)
    n = args.num_picture
    fake_imgs = sample_unconditional_model(model, n * n)
    if save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for index in range(n * n):
            RGB = np.zeros((img_size, img_size, 3))
            RGB[:, :, 0] = fake_imgs[index][2]
            RGB[:, :, 1] = fake_imgs[index][1]
            RGB[:, :, 2] = fake_imgs[index][0]
            RGB = 255 * (RGB - RGB.min()) / (RGB.max() - RGB.min())
            RGB = RGB.astype('uint8')
            RGB = cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR)
            img_name = os.path.join(save_path, f'{index}.jpg')
            cv2.imwrite(img_name, RGB)
    else:
        index = 0
        RGB = np.zeros((img_size, img_size, 3))
        RGB[:, :, 0] = fake_imgs[index][2]
        RGB[:, :, 1] = fake_imgs[index][1]
        RGB[:, :, 2] = fake_imgs[index][0]

        RGB = 255 * (RGB - RGB.min()) / (RGB.max() - RGB.min())
        RGB = RGB.astype('uint8')
        plt.imshow(RGB)
        plt.show()


if __name__ == '__main__':
    main()
