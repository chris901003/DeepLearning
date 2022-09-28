import os
from tqdm import tqdm


def main():
    video_path = r'C:\Dataset\PoseTest'
    assert os.path.isdir(video_path), '傳入的需要是資料夾位置'
    support_video_format = ['.mp4', '.MOV']
    labels = [folder_name for folder_name in os.listdir(video_path)
              if os.path.isdir(os.path.join(video_path, folder_name))]
    results = list()
    for label in tqdm(labels):
        folder_path = os.path.join(video_path, label)
        for video_name in os.listdir(folder_path):
            if os.path.splitext(video_name)[1] not in support_video_format:
                continue
            video = os.path.join(video_path, label, video_name)
            res = video + ' ' + label
            results.append(res)
    save_path = os.path.join(video_path, 'annotations.txt')
    with open(save_path, 'w') as f:
        for result in results:
            f.write(result)
            f.write('\n')
    print('Finish')


if __name__ == '__main__':
    main()
