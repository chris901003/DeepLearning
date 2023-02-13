import tensorrt
import argparse
import numpy as np
import cv2
import time
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.WorkingFlow.build import WorkingSequence


def parse_args():
    parser = argparse.ArgumentParser()
    # 工作流程設定檔案資料
    parser.add_argument('--WorkingFlowCfgPath', type=str,
                        default=r'C:\DeepLearning\SpecialTopic\Verify\EatingTime\working_flow_cfg.json')
    # 結果保存位置
    parser.add_argument('--ResultSavePath', type=str,
                        default=r'C:\DeepLearning\SpecialTopic\Verify\EatingTime\ResultSave\remain_time')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    working_flow_cfg_path = args.WorkingFlowCfgPath
    result_save_path = args.ResultSavePath
    pTime = time.time()
    working_flow_cfg = parser_cfg(working_flow_cfg_path)
    working_flow = WorkingSequence(working_flow_cfg)
    step_add_input = {'ObjectClassifyToRemainClassify': {'0': {'using_dict_name': 'FoodDetection9'}}}
    time_record_list = list()
    while True:
        result = working_flow(step_add_input=step_add_input)
        image_info = result.get('image', None)
        assert image_info is not None, '圖像資料消失，請檢查異常'
        tracking_object = result.get('track_object_info', None)
        assert tracking_object is not None, '給定流程有問題'
        rgb_image = image_info.get('rgb_image')
        deep_color_image = image_info.get('deep_draw')
        if np.min(rgb_image) == 0 and np.max(rgb_image) == 0:
            break
        if len(tracking_object) != 0:
            # 從working flow中的輸出就會有預估的剩餘時間以及當前畫面上碼錶的時間
            assert len(tracking_object) == 1, '追蹤對象超出一個，請將環境清理乾淨'
            tracking_object = tracking_object[0]
            predict_remain_time = tracking_object.get('remain_time', None)
            assert predict_remain_time is not None, '無法取得remain_time資訊，請查看過程是否有誤'
            # 這裡我將計時器的時間資料的key設定成stopwatch_time
            stopwatch_time = tracking_object.get('stopwatch_time', None)
            assert stopwatch_time is not None, '無法取得碼表時間，請查看名稱是否有誤'
            cv2.putText(rgb_image, f"Remain Time : {predict_remain_time}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(rgb_image, f"Stopwatch Time : {stopwatch_time}",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # 這裡的時間都是以秒為單位
            # 理論上predict_remain_time的值會越來越小，stopwatch_time的值會越來越大
            if isinstance(predict_remain_time, str):
                predict_remain_time = -1
            time_record_list.append(dict(predict_remain_time=predict_remain_time, stopwatch_time=stopwatch_time))

            # 框出目標位置
            position = tracking_object.get('position', None)
            assert position is not None, '需要提供目標座標位置'
            track_id = tracking_object.get('track_id', None)
            assert track_id is not None, '須提供追蹤ID'
            remain = tracking_object.get('category_from_remain', None)
            assert remain is not None, '須提供剩餘量資料'
            if isinstance(remain, float):
                remain = round(remain, 2)
            image_height, image_width = rgb_image.shape[:2]
            xmin, ymin, xmax, ymax = position.astype(np.int32)
            xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(image_width, xmax), min(image_height, ymax)
            cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(rgb_image, f"Track ID : {track_id}",
                        (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(rgb_image, f"Remain : {remain}",
                        (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 顯示碼表資料
            stopwatch_detail = tracking_object.get('stopwatch_detail', None)
            if stopwatch_detail is not None:
                stopwatch_labels = stopwatch_detail.get('stopwatch_labels', None)
                stopwatch_boxes = stopwatch_detail.get('stopwatch_boxes', None)
                assert stopwatch_labels is not None and stopwatch_boxes is not None, \
                    '在stopwatch_detail中需要提供stopwatch_labels與stopwatch_boxes資料'
                for label, box in zip(stopwatch_labels, stopwatch_boxes):
                    ymin, xmin, ymax, xmax = box
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(image_width, xmax), min(image_height, ymax)
                    cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                    cv2.putText(rgb_image, f'{label}',
                                (xmax, ymax + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(rgb_image, f"FPS : {int(fps)}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('RGB Video', rgb_image)
        cv2.imshow('Deep Color Video', deep_color_image)
        if cv2.waitKey(1) == ord('q'):
            break
    time_record = np.array(time_record_list)
    np.save(result_save_path, time_record)


if __name__ == '__main__':
    main()
