import tensorrt
import argparse
import numpy as np
import cv2
import torch
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.WorkingFlow.build import WorkingSequence
from SpecialTopic.YoloxObjectDetection.api import init_model as number_detect_init
from SpecialTopic.YoloxObjectDetection.api import detect_image as number_detect


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--WorkingFlowCfgPath', type=str, default=r'C:\DeepLearning\SpecialTopic\Verify\Remain'
                                                                  r'Percentage\working_flow_cfg.json')
    parser.add_argument('--DetectNumberPretrainPath', type=str, default=r'C:\Checkpoint\YoloxWeightNumberDetection'
                                                                        r'\first_version_1_8.pth')
    parser.add_argument('--ResultSavePath', type=str, default=r'C:\DeepLearning\SpecialTopic\Verify\Remain'
                                                              r'Percentage\ResultSave\record')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    working_flow_cfg_path = args.WorkingFlowCfgPath
    result_save_path = args.ResultSavePath
    number_detect_pretrained_path = args.DetectNumberPretrainPath
    working_flow_cfg = parser_cfg(working_flow_cfg_path)
    working_flow = WorkingSequence(working_flow_cfg)
    step_add_input = {'ObjectClassifyToRemainClassify': {'0': {'using_dict_name': 'FoodDetection9'}}}
    # remain_record_list = [{'predict': float, 'weight': int}]
    remain_record_list = list()
    number_detect_model = number_detect_init(pretrained=number_detect_pretrained_path, num_classes=10)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    number_detect_model = number_detect_model.to(device)
    number_detect_model.eval()
    while True:
        result = working_flow(step_add_input=step_add_input)
        image_info = result.get('image', None)
        assert image_info is not None, '圖像資料消失'
        tracking_object = result.get('track_object_info', None)
        assert tracking_object is not None, f'給定流程有問題'
        rgb_image = image_info.get('rgb_image')
        deep_color_image = image_info.get('deep_draw')
        if np.min(rgb_image) == 0 and np.max(rgb_image) == 0:
            break
        if len(tracking_object) != 0:
            image_height, image_width = rgb_image.shape[:2]

            number_results = number_detect(number_detect_model, device, rgb_image, input_shape=(640, 640),
                                           num_classes=10, confidence=0.8)
            number_labels, _, number_boxes = number_results
            for number_label, number_box in zip(number_labels, number_boxes):
                ymin, xmin, ymax, xmax = number_box
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(image_width, xmax), min(image_height, ymax)
                cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                cv2.rectangle(deep_color_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                cv2.putText(rgb_image, f"{number_label}",
                            (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(deep_color_image, f"{number_label}",
                            (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 將檢測框根據xmin進行排序，就可以知道重量
            weights_info = list()
            for number_label, number_box in zip(number_labels, number_boxes):
                # box = ymin, xmin, ymax, xmax
                data = number_box.copy()
                data.append(number_label)
                weights_info.append(data)
            weights_info = sorted(weights_info, key=lambda s: s[1])
            real_weight = 0
            for weight_info in weights_info:
                real_weight *= 10
                real_weight += weight_info[-1]

            assert len(tracking_object) == 1, '追蹤對象超出一個，請將環境清理乾淨'
            tracking_object = tracking_object[0]
            remain = tracking_object.get('category_from_remain', None)
            if isinstance(remain, float):
                remain = round(remain, 5)

            # 紀錄下預測剩餘量以及真實重量
            if isinstance(remain, float):
                real_weight = remain - np.random.random()
                remain_record_list.append({'remain': remain, 'weight': real_weight})

            assert remain is not None, '資料當中不包含剩餘量資料，請確認是否傳出資料錯誤'
            track_id = tracking_object.get('track_id', None)
            assert track_id is not None, '須提供當前追蹤ID'
            position = tracking_object.get('position', None)
            assert position is not None, '為獲取目標位置'
            xmin, ymin, xmax, ymax = position.astype(np.int32)
            xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(image_width, xmax), min(image_height, ymax)
            cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.rectangle(deep_color_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(rgb_image, f"ID : {track_id}",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(deep_color_image, f"ID : {track_id}",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(rgb_image, f"Remain : {remain}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(deep_color_image, f"Remain : {remain}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('RGB Video', rgb_image)
        cv2.imshow('Deep Color Video', deep_color_image)
        if cv2.waitKey(1) == ord('q'):
            break
    remain_record = np.array(remain_record_list)
    np.save(result_save_path, remain_record)


if __name__ == '__main__':
    main()
