# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import mmcv

from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def main():
    """Visualize the demo images."""
    # 關節點檢測展示傳入參數
    parser = ArgumentParser()
    # 傳入指定模型設定資料
    parser.add_argument('pose_config', help='Config file for detection')
    # 訓練權重模型檔案路徑
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    # 要進行預測的圖像資料，可以是單張圖像或是一個資料夾
    parser.add_argument(
        '--img-path',
        type=str,
        help='Path to an image file or a image folder.')
    # 是否將圖像進行展示
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    # 預測結果標註圖像檔案輸出位置
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    # 進行推理設備
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    # 關節點閾值設定
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    # OKS的閾值設定給NMS使用
    parser.add_argument(
        '--pose-nms-thr',
        type=float,
        default=0.9,
        help='OKS threshold for pose NMS')
    # 關節點標註，會在點上進行畫圓，指定圓的半徑
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    # 指定關節點中間畫線的粗度
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    # 將參數進行包裝
    args = parser.parse_args()

    # 只少需要設定show或是out_img_root否則最終結果會直接丟到垃圾桶
    assert args.show or (args.out_img_root != '')

    # prepare image list
    if osp.isfile(args.img_path):
        # 如果img_path是單一檔案就會到這裡，直接使用list進行包裝
        image_list = [args.img_path]
    elif osp.isdir(args.img_path):
        # 如果img_path是資料夾就會到這裡，將資料夾當中符合以下副檔名的就會存到list當中
        image_list = [
            osp.join(args.img_path, fn) for fn in os.listdir(args.img_path)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ]
    else:
        # 其他情況就會直接報錯
        raise ValueError('Image path should be an image or image folder.'
                         f'Got invalid image path: {args.img_path}')

    # build the pose model from a config file and a checkpoint file
    # 根據傳入的模型config文件進行構建模型，同時會加載模型權重
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device.lower())

    # 獲取dataset的type資訊
    dataset = pose_model.cfg.data['test']['type']
    # 獲取test模式下的dataset_info資訊
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        # 如果沒有設定dataset_info就會到這裡跳出警告
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # 這裡dataset需要是BottomUpCocoDataset否則就會報錯
        assert (dataset == 'BottomUpCocoDataset')
    else:
        # 構建DatasetInfo實例化對象
        dataset_info = DatasetInfo(dataset_info)

    # optional
    # 是否需要將熱力圖返回，這裡可以選擇成True
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    # 如果有想要獲取中間層輸出就可以在這裡將需要的輸出寫上
    output_layer_names = None

    # process each image
    # 遍歷每張圖像，將圖像進行加工處理
    for image_name in mmcv.track_iter_progress(image_list):
        # image_name = 圖像資料檔案路徑

        # test a single image, with a list of bboxes.
        # 使用inference_bottom_up_pose_model進行單張圖像預測
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            image_name,
            dataset=dataset,
            dataset_info=dataset_info,
            pose_nms_thr=args.pose_nms_thr,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(
                args.out_img_root,
                f'vis_{osp.splitext(osp.basename(image_name))[0]}.jpg')

        # show the results
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=args.show,
            out_file=out_file)


if __name__ == '__main__':
    main()
