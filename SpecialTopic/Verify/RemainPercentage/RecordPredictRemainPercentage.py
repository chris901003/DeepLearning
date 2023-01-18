import tensorrt
import argparse
import numpy as np
import cv2
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.WorkingFlow.build import WorkingSequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--WorkingFlowCfgPath', type=str, default=r'C:\DeepLearning\SpecialTopic\Verify\Remain'
                                                                  r'Percentage\working_flow_cfg.json')
    parser.add_argument('--ResultSavePath', type=str, default=r'C:\DeepLearning\SpecialTopic\Verify\Remain'
                                                              r'Percentage\ResultSave\record')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    working_flow_cfg_path = args.WorkingFlowCfgPath
    result_save_path = args.ResultSavePath
    working_flow_cfg = parser_cfg(working_flow_cfg_path)
    working_flow = WorkingSequence(working_flow_cfg)
    step_add_input = {'ObjectClassifyToRemainClassify': {'0': {'using_dict_name': 'FoodDetection9'}}}
    remain_record_list = list()
    while True:
        result = working_flow(step_add_input=step_add_input)
        image_info = result.get('image', None)
        assert image_info is not None, '圖像資料消失'
        tracking_object = result.get('track_object_info', None)
        assert tracking_object is not None, f'給定流程有問題'
        rgb_image = image_info.get('rgb_image')
        deep_color_image = image_info.get('deep_draw')
        if len(tracking_object) != 0:
            assert len(tracking_object) == 1, '追蹤對象超出一個，請將環境清理乾淨'
            tracking_object = tracking_object[0]
            image_height, image_width = rgb_image.shape[:2]
            remain = tracking_object.get('category_from_remain', None)
            if isinstance(remain, float):
                remain = round(remain, 5)
                remain_record_list.append(remain)
            assert remain is not None, '資料當中不包含剩餘量資料，請確認是否傳出資料錯誤'
            position = tracking_object.get('position', None)
            assert position is not None, '為獲取目標位置'
            xmin, ymin, xmax, ymax = position.astype(np.int)
            xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(image_width, xmax), min(image_height, ymax)
            cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.rectangle(deep_color_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
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
