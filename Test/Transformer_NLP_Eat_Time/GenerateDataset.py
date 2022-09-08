import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


def get_close(pass_time, interval):
    pass_time = math.ceil(pass_time / interval) * interval
    return pass_time


def main():
    num_line = 2000
    min_time = 20
    max_time = 60
    interval = 1
    draw_line = True
    save_path = './data'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    assert os.path.isdir(save_path), '保存資料位置需要是資料夾'
    lines = []
    for idx in tqdm(range(num_line)):
        pass_time = np.random.randint(low=min_time, high=max_time + 1)
        pass_time = get_close(pass_time, interval)
        num_points = pass_time // interval
        assert num_points >= 1, '採樣點至少需要大於1'
        points = np.random.randint(low=1, high=100, size=(num_points - 1,))
        points = np.append(points, [0, 100])
        points = np.sort(points, axis=0)[::-1]
        total_points = len(points) - 1
        time = [idx for idx in range(total_points, -1, -1)]
        # time = [index for index in range(0, pass_time + 1, interval)]
        left = points.tolist()
        cut_point = np.random.randint(low=3, high=len(left) + 1)
        if np.random.random() > 0.3:
            time = time[:cut_point]
            left = left[:cut_point]
        line = plt.plot(time, left)
        plt.gca().invert_xaxis()
        plt.title('Time-Food')
        plt.ylabel('Food')
        plt.xlabel('Time')
        plt.setp(line, marker='o')
        plt.grid(True)
        plt.text(x=time[0], y=100, s=left, wrap=True)
        if draw_line:
            save_img = os.path.join(save_path, f'{idx}.png')
            plt.savefig(save_img)
            plt.cla()
        data = {
            'time': time,
            'left': left
        }
        lines.append(data)
    save_pkl = os.path.join(save_path, 'food_time.pkl')
    with open(save_pkl, 'wb') as f:
        pickle.dump(lines, f)


if __name__ == '__main__':
    main()
