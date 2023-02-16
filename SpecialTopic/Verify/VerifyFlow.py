import tensorrt
import argparse
import os
import json
import cv2
import torch
from tqdm import tqdm
import time
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as font
from SpecialTopic.WorkingFlow.build import WorkingSequence
from SpecialTopic.YoloxObjectDetection.api import init_model as object_detection_init
from SpecialTopic.YoloxObjectDetection.api import detect_image as object_detection_detect


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser()
    # 共通參數
    # 預測時的工作流設定(基本上不用變更)
    parser.add_argument('--working-flow-cfg-path', type=str,
                        default=r'C:\DeepLearning\SpecialTopic\Verify\EatingTime\working_flow_cfg.json')
    # 影片保存路徑
    parser.add_argument('--video-save-path', type=str,
                        default=r'C:\DeepLearning\SpecialTopic\Verify\VideoSave\Test')
    # 結果保存根目錄位置(基本上這裡不用更改)
    parser.add_argument('--result-save-root', type=str, default=r'C:\DeepLearning\SpecialTopic\Verify\Result')
    # 結果保存在根目錄下的資料夾名稱
    parser.add_argument('--result-save-folder-name', type=str, default='Test')

    # 剩餘量驗證參數
    # 第一部分參數(基本上不用改)
    parser.add_argument('--detect-number-pretrain-path', type=str,
                        default=r'C:\Checkpoint\YoloxWeightNumberDetection\weight_number.pth')
    # 第二部分參數
    parser.add_argument('--remain-num-part', type=int, default=4)

    # 剩餘時間參數
    # 第一部分參數(基本上不用改)
    parser.add_argument('--detect-time-pretrain-path', type=str,
                        default=r'C:\Checkpoint\YoloxWeightNumberDetection\stopwatch.pth')
    # 第二部分參數
    parser.add_argument('--time-num-part', type=int, default=4)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # 提取出參數資料，並且進行初始化
    working_flow_cfg_path = args.working_flow_cfg_path
    video_save_path = args.video_save_path
    result_save_root = args.result_save_root
    result_save_folder_name = args.result_save_folder_name
    detect_number_pretrain_path = args.detect_number_pretrain_path
    remain_num_part = args.remain_num_part
    detect_time_pretrain_path = args.detect_time_pretrain_path
    time_num_part = args.time_num_part
    assert os.path.exists(working_flow_cfg_path)
    assert os.path.exists(video_save_path)
    assert os.path.exists(detect_number_pretrain_path)
    assert os.path.exists(detect_time_pretrain_path)
    result_save_folder_path = os.path.join(result_save_root, result_save_folder_name)
    if not os.path.exists(result_save_root):
        os.mkdir(result_save_root)
    if not os.path.exists(result_save_folder_path):
        os.mkdir(result_save_folder_path)

    # 獲取秤重機以及碼表路徑
    rgb_video_path = os.path.join(video_save_path, 'RgbView.avi')
    assert os.path.exists(rgb_video_path), '須提供彩色影片'
    stopwatch_video_path = os.path.join(video_save_path, 'Stopwatch.mp4')
    assert os.path.exists(stopwatch_video_path), '須提供碼表的影片'
    weight_video_path = os.path.join(video_save_path, 'Weight.mp4')
    assert os.path.exists(weight_video_path), '須提供秤重機的影片'

    # 將碼表影片合成到影片上
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    stopwatch_cap = cv2.VideoCapture(stopwatch_video_path)
    fps = int(rgb_cap.get(cv2.CAP_PROP_FPS))
    width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rgb_mix_stopwatch_video_path = os.path.join(video_save_path, 'MixStopwatch.avi')
    rgb_mix_stopwatch_writer = cv2.VideoWriter(rgb_mix_stopwatch_video_path, cv2.VideoWriter_fourcc(*'XVID'),
                                               fps, (width, height), 1)
    if rgb_cap.get(7) != stopwatch_cap.get(7):
        raise RuntimeError('彩色影片與碼表影片長度不同')
    total_frame = rgb_cap.get(7)
    progress = tqdm(total=total_frame)
    time_height = 0
    time_width = 0
    while True:
        rgb_ret, rgb_image = rgb_cap.read()
        time_ret, time_image = stopwatch_cap.read()
        if rgb_ret and time_ret:
            time_height, time_width = time_image.shape[:2]
            rgb_image[0:time_height, 0:time_width, :3] = time_image
            rgb_mix_stopwatch_writer.write(rgb_image)
            progress.update(1)
        else:
            progress.close()
            break
    rgb_mix_stopwatch_writer.release()

    # 修改讀彩色影片以及深度資料的設定
    working_flow_cfg = parse_json(working_flow_cfg_path)
    read_picture_cfg_path = working_flow_cfg['step1']['config_file']
    read_picture_cfg = parse_json(read_picture_cfg_path)
    read_picture_data_cfg_path = read_picture_cfg['rgbd_record_config_path']
    read_picture_data_info = parse_json(read_picture_data_cfg_path)
    read_picture_data_info['rgb_path'] = rgb_mix_stopwatch_video_path
    read_picture_data_info['deep_path'] = video_save_path
    rewrite_cfg(read_picture_data_cfg_path, read_picture_data_info)

    # 實例化預測流，並且將追蹤時間拉長
    working_flow = WorkingSequence(working_flow_cfg=working_flow_cfg)
    working_flow.steps[1]['module'].module.tracking_keep_period = 10000
    working_flow.steps[1]['module'].module.mod_frame_index = 10000 * 10
    working_flow.steps[3]['module'].module.save_last_period = 10000
    working_flow.steps[3]['module'].module.mod_frame = 10000 * 10

    # 碼表在畫面上的時間，推理當前碼表數字時只會使用有效的位置
    working_flow.steps[4]['module'].module.screen_xmin = 0
    working_flow.steps[4]['module'].module.screen_ymin = 0
    working_flow.steps[4]['module'].module.screen_width = time_width
    working_flow.steps[4]['module'].module.screen_height = time_height

    # 一些其他設定，以及保存位置
    weight_cap = cv2.VideoCapture(weight_video_path)
    step_add_input = {'ObjectClassifyToRemainClassify': {'0': {'using_dict_name': 'FoodDetection9'}}}
    remain_record_list = list()
    remain_time_record_list = list()
    pTime = time.time()

    # 設定重量目標檢測模型
    number_detect_model = object_detection_init(pretrained=detect_number_pretrain_path, num_classes=10)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    number_detect_model = number_detect_model.to(device)
    number_detect_model.eval()

    # 開始對影片進行解析
    while True:
        weight_ret, weight_image = weight_cap.read()
        result = working_flow(step_add_input=step_add_input)
        image_info = result.get('image', None)
        assert image_info is not None, '圖像資料丟失'
        tracking_object = result.get('track_object_info', None)
        assert tracking_object is not None, '給定流程有問題'
        rgb_image = image_info.get('rgb_image', None)
        deep_color_image = image_info.get('deep_draw', None)
        assert rgb_image is not None and deep_color_image is not None
        if np.min(rgb_image) == 0 and np.max(rgb_image) == 0:
            # 當彩色圖全為0時表示影片讀取完畢
            if weight_ret:
                raise RuntimeError('理論上秤重機資訊長度應為依樣長')
            else:
                break
        weight_height, weight_width = weight_image.shape[:2]
        if len(tracking_object) != 0:
            # 檢測當前秤重機數字
            number_results = object_detection_detect(number_detect_model, device, weight_image, input_shape=(640, 640),
                                                     num_classes=10, confidence=0.8)
            number_labels, _, number_boxes = number_results
            weights_info = list()
            for number_label, number_box in zip(number_labels, number_boxes):
                data = number_box.copy()
                data.append(number_label)
                weights_info.append(data)
            weights_info = sorted(weights_info, key=lambda s: s[1])
            real_weight = 0
            for weight_info in weights_info:
                real_weight *= 10
                real_weight += weight_info[-1]

            # 獲取預估資料，並且保存下來
            assert len(tracking_object) == 1, '追蹤對象超過一個，需要將環境清空才可以驗證'
            tracking_object = tracking_object[0]
            remain = tracking_object.get('category_from_remain', None)
            assert remain is not None, '須提供預估剩餘量'
            if isinstance(remain, float):
                remain = round(remain, 5)
            predict_remain_time = tracking_object.get('remain_time', None)
            assert predict_remain_time is not None, '須提供預估剩餘時間'
            stopwatch_time = tracking_object.get('stopwatch_time', None)
            assert stopwatch_time is not None, '須提供碼表時間'
            if not isinstance(predict_remain_time, str):
                remain_record_list.append({'predict_remain': remain, 'weight': real_weight,
                                           'stopwatch_time': stopwatch_time})
                remain_time_record_list.append({'predict_remain_time': predict_remain_time,
                                                'stopwatch_time': stopwatch_time})

            # 將資料畫到畫面上，並且顯示出來
            track_id = tracking_object.get('track_id', None)
            assert track_id is not None
            position = tracking_object.get('position', None)
            assert position is not None
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            xmin, ymin, xmax, ymax = position.astype(np.int32)
            xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(width, xmax), min(height, ymax)
            cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(rgb_image, f"ID : {track_id}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(rgb_image, f"Remain : {remain}",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(rgb_image, f"Remain Time : {predict_remain_time}",
                        (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(rgb_image, f"Stopwatch Time : {stopwatch_time}",
                        (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(rgb_image, f"Weight : {real_weight}",
                        (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(weight_image, f"Weight : {real_weight}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(rgb_image, f"FPS : {int(fps)}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 最後顯示出來
        cv2.imshow('RGB Video', rgb_image)
        cv2.imshow('Deep Color Video', deep_color_image)
        cv2.namedWindow("Weight", 0)
        cv2.resizeWindow("Weight", weight_width, weight_height)
        cv2.imshow('Weight', weight_image)
        if cv2.waitKey(1) == ord('q'):
            break

    # 保存原始資料
    remain_record_save_path = os.path.join(result_save_folder_path, 'remain_record')
    remain_time_record_save_path = os.path.join(result_save_folder_path, 'remain_time_record')
    remain_record = np.array(remain_record_list)
    remain_time_record = np.array(remain_time_record_list)
    np.save(remain_record_save_path, remain_record)
    np.save(remain_time_record_save_path, remain_time_record)

    # 將預測完的結果進行驗證
    assert remain_num_part > 0, '須至少大於一段'
    assert time_num_part > 0, '須至少大於一段'
    assert len(remain_record_list) > remain_num_part, '影片長度不夠'
    assert len(remain_time_record_list) > time_num_part, '影片長度不夠'

    # 壓縮資料
    remain_record_list = compress_remain(remain_record_list)
    remain_time_record_list = compress_remain_time(remain_time_record_list)
    assert len(remain_record_list) == len(remain_time_record_list), '長度理論上要相同'
    total_time = len(remain_record_list)

    # 將重量轉換成真實剩餘量以及將碼表資訊轉成真實剩餘時間
    remain_record_list = transfer_weight_to_real_remain(remain_record_list)
    remain_time_record_list = transfer_stopwatch_to_real_remain_time(remain_time_record_list)

    # 計算出每一秒的損失值
    remain_record_loss = remain_cal_loss(remain_record_list)
    remain_time_record_loss = remain_time_cal_loss(remain_time_record_list)

    # 根據段數，獲取每段的平均誤差
    remain_part_time = math.ceil(total_time / remain_num_part)
    remain_time_part_time = math.ceil(total_time / time_num_part)
    remain_record_loss = cal_part_loss(remain_record_loss, remain_part_time)
    remain_time_record_loss = cal_part_loss(remain_time_record_loss, remain_time_part_time)

    # 後面畫圖需要用到的資料
    remain_time_line = [remain_part_time * i for i in range(remain_num_part + 1)]
    remain_time_line[-1] = min(remain_time_line[-1], total_time)
    remain_time_line = parse_time_line(remain_time_line)
    remain_time_time_line = [remain_time_part_time * i for i in range(time_num_part + 1)]
    remain_time_time_line[-1] = min(remain_time_time_line[-1], total_time)
    remain_time_time_line = parse_time_line(remain_time_time_line)

    # 獲取原始資料
    predict_remain_list = list()
    real_remain_list = list()
    for info in remain_record_list:
        predict = info.get('predict_remain', None)
        real = info.get('real_remain', None)
        assert predict is not None and real is not None
        predict_remain_list.append(predict)
        real_remain_list.append(real)
    predict_remain_time_list = list()
    real_remain_time_list = list()
    for info in remain_time_record_list:
        predict = info.get('predict_remain_time', None)
        real = info.get('real_remain_time', None)
        assert predict is not None and real is not None
        predict_remain_time_list.append(predict)
        real_remain_time_list.append(real)

    # 保存原始數據到Json檔案中
    raw_json_save_path = os.path.join(result_save_folder_path, 'raw_info.json')
    raw_save_info = dict(
        predict_remain=predict_remain_list,
        real_remain=real_remain_list,
        predict_remain_time=predict_remain_time_list,
        real_remain_time=real_remain_time_list,
        remain_l1_loss_sec=remain_record_loss['l1_loss'],
        remain_l2_loss_sec=remain_record_loss['l2_loss'],
        remain_time_l1_loss_sec=remain_time_record_loss['l1_loss'],
        remain_time_l2_loss_sec=remain_time_record_loss['l2_loss'],
        remain_l1_loss_part=remain_record_loss['l1_part'],
        remain_l2_loss_part=remain_record_loss['l2_part'],
        remain_time_l1_loss_part=remain_time_record_loss['l1_part'],
        remain_time_l2_loss_part=remain_time_record_loss['l2_part'],
        remain_l1_avg=remain_record_loss['avg_l1'],
        remain_l2_avg=remain_record_loss['avg_l2'],
        remain_time_l1_avg=remain_time_record_loss['avg_l1'],
        remain_time_l2_avg=remain_time_record_loss['avg_l2'],
    )
    with open(raw_json_save_path, 'w') as f:
        json.dump(raw_save_info, f, indent=4)

    # 畫圖表
    font1 = font(fname=r'C:\DeepLearning\SpecialTopic\Verify\EatingTime\NotoSansTC-Bold.otf')
    # 剩餘量圖表
    _ = plt.figure(figsize=(11, 7))
    plt.subplot(611)
    plt.title('剩餘時間', fontproperties=font1)
    plt.plot(predict_remain_list, 'b--', label='預估')
    plt.plot(real_remain_list, 'r-', label='真實')
    plt.legend(loc='best', prop=font1)
    plt.subplot(612)
    plt.title('L1', fontproperties=font1)
    plt.plot(remain_record_loss['l1_loss'], 'b-')
    plt.subplot(613)
    plt.title('L2', fontproperties=font1)
    plt.plot(remain_record_loss['l2_loss'], 'b-')
    plt.subplot(614)
    plt.title('L1 分段', fontproperties=font1)
    plt.bar(x=range(len(remain_record_loss['l1_part'])), height=remain_record_loss['l1_part'],
            tick_label=remain_time_line, color='g')
    plt.subplot(615)
    plt.title('L2 分段', fontproperties=font1)
    plt.bar(x=range(len(remain_record_loss['l2_part'])), height=remain_record_loss['l2_part'],
            tick_label=remain_time_line, color='r')
    plt.subplot(616)
    plt.title('總平均', fontproperties=font1)
    plt.text(0, 0.5, f"L1 Avg: {remain_record_loss['avg_l1']}", fontsize=15, color='blue')
    plt.text(0, 0, f"L2 Avg: {remain_record_loss['avg_l2']}", fontsize=15, color='blue')
    plt.tight_layout()
    remain_chart_save_path = os.path.join(result_save_folder_path, 'remain_chart.jpg')
    plt.savefig(remain_chart_save_path)
    plt.show()

    # 剩餘時間圖表
    _ = plt.figure(figsize=(11, 7))
    plt.subplot(611)
    plt.title('剩餘時間', fontproperties=font1)
    plt.plot(predict_remain_time_list, 'b--', label='預估')
    plt.plot(real_remain_time_list, 'r-', label='真實')
    plt.legend(loc='best', prop=font1)
    plt.subplot(612)
    plt.title('L1', fontproperties=font1)
    plt.plot(remain_time_record_loss['l1_loss'], 'b-')
    plt.subplot(613)
    plt.title('L2', fontproperties=font1)
    plt.plot(remain_time_record_loss['l2_loss'], 'b-')
    plt.subplot(614)
    plt.title('L1 分段', fontproperties=font1)
    plt.bar(x=range(len(remain_time_record_loss['l1_part'])), height=remain_time_record_loss['l1_part'],
            tick_label=remain_time_time_line, color='g')
    plt.subplot(615)
    plt.title('L2 分段', fontproperties=font1)
    plt.bar(x=range(len(remain_time_record_loss['l2_part'])), height=remain_time_record_loss['l2_part'],
            tick_label=remain_time_time_line, color='r')
    plt.subplot(616)
    plt.title('總平均', fontproperties=font1)
    plt.text(0, 0.5, f"L1 Avg: {remain_time_record_loss['avg_l1']}", fontsize=15, color='blue')
    plt.text(0, 0, f"L2 Avg: {remain_time_record_loss['avg_l2']}", fontsize=15, color='blue')
    plt.tight_layout()
    remain_time_chart_save_path = os.path.join(result_save_folder_path, 'remain_time_chart.jpg')
    plt.savefig(remain_time_chart_save_path)
    plt.show()


def parse_json(file_path):
    # 解析json資料
    with open(file_path, 'r') as f:
        info = json.load(f)
    return info


def rewrite_cfg(file_path, file_info):
    # 重寫json檔案
    with open(file_path, 'w') as f:
        json.dump(file_info, f, indent=4)


def compress_remain(remain_record_list):
    """ 將剩餘量資料進行壓縮，將一秒內的資料進行平均 """
    results = list()
    last_stopwatch_time = remain_record_list[0].get('stopwatch_time', None)
    assert last_stopwatch_time is not None, '須提供stopwatch_time資訊才可以進行時間壓縮'
    tmp_remain_list = list()
    tmp_weight_list = list()
    for info in remain_record_list:
        stopwatch_time = info.get('stopwatch_time', None)
        predict_remain = info.get('predict_remain', None)
        weight = info.get('weight', None)
        assert stopwatch_time is not None and predict_remain is not None and weight is not None, \
            'remain_record_list有缺少資料'
        if stopwatch_time == last_stopwatch_time:
            tmp_remain_list.append(predict_remain)
            tmp_weight_list.append(weight)
        else:
            avg_remain = sum(tmp_remain_list) / len(tmp_remain_list)
            avg_weight = sum(tmp_weight_list) / len(tmp_weight_list)
            data = dict(stopwatch_time=last_stopwatch_time, predict_remain=avg_remain, weight=avg_weight)
            results.append(data)
            tmp_remain_list = [predict_remain]
            tmp_weight_list = [weight]
            last_stopwatch_time = stopwatch_time
    if len(tmp_remain_list) != 0:
        avg_remain = sum(tmp_remain_list) / len(tmp_remain_list)
        avg_weight = sum(tmp_weight_list) / len(tmp_weight_list)
        data = dict(stopwatch_time=last_stopwatch_time, predict_remain=avg_remain, weight=avg_weight)
        results.append(data)
    return results


def compress_remain_time(remain_time_record_list):
    """ 將剩餘量進行壓縮，將一秒內的資料取平均"""
    results = list()
    last_stopwatch_time = remain_time_record_list[0].get('stopwatch_time', None)
    tmp_predict_remain_time = list()
    assert last_stopwatch_time is not None, '須提供stopwatch_time資訊'
    for info in remain_time_record_list:
        predict_remain_time = info.get('predict_remain_time', None)
        stopwatch_time = info.get('stopwatch_time', None)
        assert predict_remain_time is not None and stopwatch_time is not None, '資料有缺少，無法進行時間壓縮'
        if last_stopwatch_time == stopwatch_time:
            tmp_predict_remain_time.append(predict_remain_time)
        else:
            avg_predict_remain_time = sum(tmp_predict_remain_time) / len(tmp_predict_remain_time)
            data = dict(predict_remain_time=avg_predict_remain_time, stopwatch_time=last_stopwatch_time)
            results.append(data)
            tmp_predict_remain_time = [predict_remain_time]
            last_stopwatch_time = stopwatch_time
    if len(tmp_predict_remain_time) != 0:
        avg_predict_remain_time = sum(tmp_predict_remain_time) / len(tmp_predict_remain_time)
        data = dict(predict_remain_time=avg_predict_remain_time, stopwatch_time=last_stopwatch_time)
        results.append(data)
    return results


def transfer_weight_to_real_remain(remain_record_list):
    """ 將重量資訊轉成剩餘量百分比
    這裡會將開始的第一幀作為重量最大值，所以在開始錄影時不要壓到秤重機
    同時最後一幀作為重量最小值，所以在要結束前不要壓到秤重機
    """
    max_weight = remain_record_list[0].get('weight', None)
    min_weight = remain_record_list[-1].get('weight', None)
    weight_range = max_weight - min_weight
    assert max_weight is not None and min_weight is not None, '無法取得秤重機資訊'
    # assert max_weight >= min_weight, '初始重量小於結束重量，資料有誤'
    print(f'最大重量: {max_weight}, 最小重量: {min_weight}')
    for idx, info in enumerate(remain_record_list):
        weight = info.get('weight', None)
        assert weight is not None
        real_remain = (weight - min_weight) / weight_range
        remain_record_list[idx]['real_remain'] = real_remain
    return remain_record_list


def transfer_stopwatch_to_real_remain_time(remain_time_record_list):
    """ 將碼表時間轉成正確剩餘時間
    會將第一幀作為開始時間，所以不要在開始錄影後很久才開始吃，否則會不準
    同時會將最後一幀做為結束時間，所以不要吃完後一直不按結束
    """
    start_time = remain_time_record_list[0].get('stopwatch_time', None)
    end_time = remain_time_record_list[-1].get('stopwatch_time', None)
    assert start_time is not None and end_time is not None, '未提供碼表資訊'
    print(f'起始秒數: {start_time}, 結束秒數: {end_time}')
    for idx, info in enumerate(remain_time_record_list):
        stopwatch_time = info.get('stopwatch_time', None)
        assert stopwatch_time is not None, '須提供stopwatch_time'
        real_remain_time = end_time - stopwatch_time
        remain_time_record_list[idx]['real_remain_time'] = real_remain_time
    return remain_time_record_list


def remain_cal_loss(remain_record_list):
    """ 剩餘量每秒的損失值 """
    l1_loss = list()
    l2_loss = list()
    for info in remain_record_list:
        predict_remain = info.get('predict_remain', None)
        real_remain = info.get('real_remain', None)
        assert predict_remain is not None and real_remain is not None, '須提供預估以及真實的剩餘量'
        l1 = abs(predict_remain - real_remain)
        l2 = (predict_remain - real_remain) * (predict_remain - real_remain)
        l1_loss.append(l1)
        l2_loss.append(l2)
    avg_l1 = sum(l1_loss) / len(l1_loss)
    avg_l2 = math.sqrt(sum(l2_loss)) / len(l2_loss)
    loss_result = dict(l1_loss=l1_loss, l2_loss=l2_loss, avg_l1=avg_l1, avg_l2=avg_l2)
    return loss_result


def remain_time_cal_loss(remain_time_record_list):
    """ 剩時間每秒的損失值 """
    l1_loss = list()
    l2_loss = list()
    for info in remain_time_record_list:
        predict_remain_time = info.get('predict_remain_time', None)
        real_remain_time = info.get('real_remain_time', None)
        assert predict_remain_time is not None and real_remain_time is not None, '須提預測剩餘時間以及真實剩餘時間'
        l1 = abs(predict_remain_time - real_remain_time)
        l2 = (predict_remain_time - real_remain_time) * (predict_remain_time - real_remain_time)
        l1_loss.append(l1)
        l2_loss.append(l2)
    avg_l1 = sum(l1_loss) / len(l1_loss)
    avg_l2 = math.sqrt(sum(l2_loss)) / len(l2_loss)
    loss_result = dict(l1_loss=l1_loss, l2_loss=l2_loss, avg_l1=avg_l1, avg_l2=avg_l2)
    return loss_result


def cal_part_loss(loss_dict, part_time):
    """ 根據指定的段數進行切割，並且獲取該段的平均誤差 """
    tmp_l1_list = list()
    tmp_l2_list = list()
    l1_part = list()
    l2_part = list()
    l1_loss = loss_dict.get('l1_loss', None)
    l2_loss = loss_dict.get('l2_loss', None)
    assert l1_loss is not None and l2_loss is not None, '須提供l1以及l2損失'
    count = 0
    for idx in range(0, len(l1_loss)):
        tmp_l1_list.append(l1_loss[idx])
        tmp_l2_list.append(l2_loss[idx])
        count += 1
        if count == part_time or (idx == len(l1_loss) - 1):
            avg_l1 = sum(tmp_l1_list) / len(tmp_l1_list)
            avg_l2 = math.sqrt(sum(tmp_l2_list)) / len(tmp_l2_list)
            l1_part.append(avg_l1)
            l2_part.append(avg_l2)
            tmp_l1_list = list()
            tmp_l2_list = list()
            count = 0
    loss_dict['l1_part'] = l1_part
    loss_dict['l2_part'] = l2_part
    return loss_dict


def parse_time_line(time_line):
    """ 處理Bar需要使用到的標籤 """
    def sec_to_min_plus_sec(sec):
        minutes = sec // 60
        sec = sec % 60
        return f"{minutes}'{sec}''"

    results = list()
    for idx in range(len(time_line) - 1):
        left_time = time_line[idx]
        right_time = time_line[idx + 1]
        left_time = sec_to_min_plus_sec(left_time)
        right_time = sec_to_min_plus_sec(right_time)
        results.append(f'{left_time}~{right_time}')
    return results


if __name__ == '__main__':
    main()
