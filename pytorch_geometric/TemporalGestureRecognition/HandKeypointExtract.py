import os
import mediapipe as mp
import torch
from tqdm import tqdm
import pickle


video_path = '/Users/huanghongyan/Documents/DeepLearning/pytorch_geometric/TemporalGestureRecognition/PoseVideo'
result_path = '/Users/huanghongyan/Documents/DeepLearning/pytorch_geometric/TemporalGestureRecognition' \
              '/PoseVideo/extract.pkl'


def main():
    assert os.path.exists(video_path), '輸入的檔案位置不存在'
    support_video_format = ['.mp4']
    video_data = list()
    for folder_name in os.listdir(video_path):
        root = os.path.join(video_path, folder_name)
        if not os.path.isdir(root):
            continue
        for video_name in os.listdir(root):
            if os.path.splitext(video_name)[1] in support_video_format:
                data = {
                    'video_path': os.path.join(root, video_name),
                    'label': folder_name
                }
                video_data.append(data)
    pipeline_cls = [ReadVideo, ExtractFrame, DecodeVideo, HandKeypointExtract, Collate]
    pipeline_args = [
        {'type': 'PyAVInit'},
        {'frames': 300, 'start': 0, 'interval': 1, 'mode': 'loop'},
        {'type': 'PyAVDecode', 'to_img': False},
        {'max_num_hands': 1, 'min_detection_confidence': 0.5, 'min_tracking_confidence': 0.5, 'idx_to_keypoint': True},
        {'targets': ['video_path', 'frames', 'keypoints', 'label', 'z_axis']}
    ]
    process = Compose(pipeline_cls, pipeline_args)
    information = {
        'keypoints': list(),
        'imgWidth': 1920,
        'imgHeight': 1080
    }
    with open(result_path, 'ab') as f:
        pickle.dump(information, f)
    for data in tqdm(video_data):
        result = process(data)
        information['keypoints'].append(result)
    with open(result_path, 'wb') as f:
        pickle.dump(information, f)
    print(f'Transfer finish total transfer {len(video_data)} videos.')


class Compose:
    def __init__(self, pipeline_cls, pipeline_args):
        assert len(pipeline_cls) == len(pipeline_args), '帶入pipeline的參數量要與pipeline的數量一樣'
        self.pipeline = list()
        for idx, operation in enumerate(pipeline_cls):
            self.pipeline.append(operation(**pipeline_args[idx]))

    def __call__(self, data):
        for operation in self.pipeline:
            data = operation(data)
        return data


class ReadVideo:
    def __init__(self, type='PyAVInit'):
        support_init = {
            'PyAVInit': PyAVInit
        }
        assert type in support_init, '該影片讀取方式未支援'
        init_function = support_init[type]
        self.read_video = init_function

    def __call__(self, data):
        data = self.read_video(data)
        return data


class ExtractFrame:
    def __init__(self, frames=300, start=0, interval=1, mode='loop'):
        support_mode = {
            'loop': ExtractFrameLoop,
            'last_frame': ExtractFrameLastFrame
        }
        assert mode in support_mode, f'{mode}尚未支援'
        self.frames = frames
        self.start = start
        self.interval = interval
        self.mode = support_mode[mode]

    def __call__(self, data):
        frame_idx = [*range(self.start, self.frames, self.interval)]
        total_frame = data.get('video_frame', None)
        assert total_frame is not None, '需獲取總幀數'
        frame_idx = self.mode(frame_idx, total_frame)
        data['frame_idx'] = frame_idx
        data['frames'] = self.frames
        return data


class DecodeVideo:
    def __init__(self, type='PyAVDecode', to_img=True):
        support_decode = {
            'PyAVDecode': PyAVDecode
        }
        assert type in support_decode, '該影片解碼方式未支援'
        self.decode_function = support_decode[type]
        self.to_img = to_img

    def __call__(self, data):
        return self.decode_function(data, self.to_img)


class HandKeypointExtract:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, show=False,
                 idx_to_keypoint=False):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=max_num_hands,
                                        min_detection_confidence=min_detection_confidence,
                                        min_tracking_confidence=min_tracking_confidence)
        self.show = show
        self.idx_to_keypoint = idx_to_keypoint

    def __call__(self, data):
        imgs = data.get('imgs', None)
        assert imgs is not None, 'data當中沒有圖像資料'
        imgHeight, imgWidth = imgs[0].shape[:2]
        keypoints = list()
        z_axis = list()
        for i, img in enumerate(imgs):
            results = self.hands.process(img)
            if results.multi_hand_landmarks:
                keypoint = list()
                z = list()
                for lm in results.multi_hand_landmarks[0].landmark:
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    # 添加上z座標，先將所有點進行標準化
                    keypoint.append([xPos, yPos])
                    z.append(lm.z)
                z = normalize_z_axis(z)
                keypoint = torch.tensor(keypoint)
                z = torch.tensor(z)
                z = z.unsqueeze(dim=-1)
                keypoints.append(keypoint)
                z_axis.append(z)
            else:
                keypoints.append(None)
                z_axis.append(None)
        if self.idx_to_keypoint:
            frame_idx = data.get('frame_idx', None)
            assert frame_idx is not None, 'Data當中缺少frame_idx參數'
            result_keypoints = [keypoints[idx] for idx in frame_idx]
            result_z_axis = [z_axis[idx] for idx in frame_idx]
        else:
            result_keypoints = keypoints
            result_z_axis = z_axis
        total_frame = len(result_keypoints)
        result_keypoints = [x for x in result_keypoints if x is not None]
        result_z_axis = [x for x in result_z_axis if x is not None]
        idx = 0
        while len(result_keypoints) < total_frame:
            result_keypoints.append(result_keypoints[idx])
            result_z_axis.append(result_z_axis[idx])
            idx += 1
        result_keypoints = torch.stack(result_keypoints)
        result_z_axis = torch.stack(result_z_axis)
        # 這裡我們先使用一隻手的關節點檢測，為了之後好擴展成多手檢測，所以先在最後添加上一個手的數量維度
        result_keypoints = result_keypoints.unsqueeze(dim=-1)
        result_z_axis = result_z_axis.unsqueeze(dim=-1)
        data['keypoints'] = result_keypoints
        data['z_axis'] = result_z_axis
        return data


class Collate:
    def __init__(self, targets):
        self.targets = targets

    def __call__(self, data):
        result = dict()
        for target in self.targets:
            info = data.get(target, None)
            assert info is not None, f'Data當中沒有{target}資訊'
            result[target] = info
        return result


def PyAVInit(data):
    try:
        import av
    except ImportError:
        raise ImportError('需要安裝PyAV才可以使用')
    video_path = data.get('video_path', None)
    assert video_path is not None, '無法獲取影片檔案位置'
    assert os.path.exists(video_path), '影片檔案不存在'
    container = av.open(video_path)
    data['video_container'] = container
    data['video_frame'] = container.streams.video[0].frames
    return data


def ExtractFrameLoop(frame_index, total_frame):
    result = [idx % total_frame for idx in frame_index]
    return result


def ExtractFrameLastFrame(frame_index, total_frame):
    result = [idx if idx < total_frame else total_frame - 1 for idx in frame_index]
    return result


def PyAVDecode(data, to_img=True):
    frame_idx = data.get('frame_idx', None)
    video_container = data.get('video_container', None)
    assert frame_idx is not None, '資料當中沒有frame_idx無法提取'
    assert video_container is not None, '需要av生成的container才可以讀取圖像'
    max_frame = max(frame_idx)
    images = list()
    for index, img in enumerate(video_container.decode(video=0)):
        images.append(img.to_rgb().to_ndarray())
        if index > max_frame:
            break
    if to_img:
        imgs = [images[index] for index in frame_idx]
        data['imgs'] = imgs
    else:
        data['imgs'] = images
    return data


def normalize_z_axis(z_axis):
    min_z = min(z_axis)
    z_axis = [z - min_z for z in z_axis]
    max_z = max(z_axis)
    z_axis = [z / max_z for z in z_axis]
    return z_axis


if __name__ == '__main__':
    main()
