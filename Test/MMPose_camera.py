import argparse
import time

import cv2
from mmpose.apis import init_pose_model
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmpose.core.post_processing import oks_nms
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.datasets.pipelines import Compose

cap = cv2.VideoCapture(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Camera Test')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument('--pose-nms-thr', type=float, default=0.9)
    parser.add_argument('--radius', type=int, default=4)
    parser.add_argument('--thickness', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    pTime = 0
    cTime = 0
    pose_model = init_pose_model(args.config, args.checkpoint, device=args.device.lower())
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)
    return_heatmap = False
    output_layer_names = None
    dataset_name = dataset_info.dataset_name
    flip_index = dataset_info.flip_index
    sigmas = getattr(dataset_info, 'sigmas', None)
    pose_results = list()
    returned_outputs = list()
    cfg = pose_model.cfg
    device = args.device.lower()
    if device == 'cpu':
        device = -1
    test_pipeline = Compose(cfg.test_pipeline)

    while True:
        ret, img = cap.read()
        if ret:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data = {'dataset': dataset_name, 'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_index': flip_index
            }, 'img': img}
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            data = scatter(data, [device])[0]
            with torch.no_grad():
                result = pose_model(
                    img=data['img'],
                    img_metas=data['img_metas'],
                    return_loss=False,
                    return_heatmap=return_heatmap
                )
            for idx, pred in enumerate(result['preds']):
                area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (np.max(pred[:, 1]) - np.min(pred[:, 1]))
                pose_results.append({
                    'keypoints': pred[:, :3],
                    'score': result['scores'][idx],
                    'area': area
                })
            score_per_joint = cfg.model.test_cfg.get('score_per_joint', False)
            keep = oks_nms(
                pose_results,
                args.pose_nms_thr,
                sigmas,
                score_per_joint=score_per_joint)
            pose_results = [pose_results[_keep] for _keep in keep]

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
