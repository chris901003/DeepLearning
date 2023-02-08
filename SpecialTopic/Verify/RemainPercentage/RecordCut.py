import argparse
import functools
import cv2
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    # 影像保存路徑，接下來回直接到指定的路徑下抓取影片以及深度資料
    parser.add_argument('--record-folder-path', '-f', type=str, default='./RgbdSave/Test')
    # 剪裁後影片資料保存資料夾
    parser.add_argument('--save-folder-path', '-p', type=str, default='./RgbdSave/TestCrop')
    # 開頭幾幀需要被移除掉
    parser.add_argument('--start-frame', '-s', type=int, default=0)
    # 結尾幾幀需要被移除掉
    parser.add_argument('--end-frame', '-e', type=int, default=0)
    args = parser.parse_args()
    return args


def file_name_comp(lhs, rhs):
    lhs_basename = os.path.basename(lhs)
    rhs_basename = os.path.basename(rhs)
    lhs_basename = os.path.splitext(lhs_basename)[0]
    rhs_basename = os.path.splitext(rhs_basename)[0]
    lhs_idx = int(lhs_basename.split('_')[1])
    rhs_idx = int(rhs_basename.split('_')[1])
    if lhs_idx < rhs_idx:
        return -1
    else:
        return 1


def main():
    args = parse_args()
    record_folder_path = args.record_folder_path
    save_folder_path = args.save_folder_path
    start_frame = args.start_frame
    end_frame = args.end_frame
    assert os.path.isdir(record_folder_path), '須給定資料夾位置'
    assert os.path.exists(record_folder_path), '給定檔案資料夾不存在'
    rgb_video_path = os.path.join(record_folder_path, 'RgbView.avi')
    assert os.path.exists(record_folder_path), f'RGB影片不存在{rgb_video_path}'
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)
    deep_info_path_list = list()
    for file_name in os.listdir(record_folder_path):
        if 'Depth_' in file_name and os.path.splitext(file_name)[1] == '.npy':
            depth_info_path = os.path.join(record_folder_path, file_name)
            deep_info_path_list.append(depth_info_path)
    deep_info_path_list.sort(key=functools.cmp_to_key(file_name_comp))

    rgb_cap = cv2.VideoCapture(rgb_video_path)
    fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_length = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = video_length - end_frame
    assert start_frame < video_length and end_frame > 0, '影片長度不夠，剪裁後將沒有任何畫面'
    current_frame = 0
    new_rgb_video_path = os.path.join(save_folder_path, 'RgbViewCrop.avi')
    color_writer = cv2.VideoWriter(new_rgb_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps,
                                   (video_width, video_height), 1)
    keep_depth_info_list = list()
    while True:
        ret, rgb_image = rgb_cap.read()
        if not ret:
            break
        if start_frame <= current_frame < end_frame:
            color_writer.write(rgb_image)
            keep_depth_info_list.append(deep_info_path_list[current_frame])
        current_frame += 1
        if current_frame == end_frame:
            break

    color_writer.release()
    for idx, source_path in enumerate(keep_depth_info_list):
        save_depth_path = os.path.join(save_folder_path, f'Depth_{idx}.npy')
        shutil.copyfile(source_path, save_depth_path)

    print(f'Original video frames: {video_length}')
    print(f'Save video frames: {len(keep_depth_info_list)}')


if __name__ == '__main__':
    main()
