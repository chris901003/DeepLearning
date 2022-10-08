import argparse
import cv2
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.WorkingFlow.build import WorkingSequence


def args_parse():
    parser = argparse.ArgumentParser('All starting from here')
    # 設定每個模塊需要接收的資料以及輸出的結果
    parser.add_argument('--working-flow-cfg', type=str,
                        default='./config/working_flow_cfg.json')
    # 接下來還會有一堆下游工作，目前先處理這兩個的連動
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    working_flow_cfg = parser_cfg(args.working_flow_cfg)
    working_flow = WorkingSequence(working_flow_cfg)
    step_add_input = {'ObjectClassifyToRemainClassify': {'0': {'using_dict_name': 'FoodDetection9'}}}
    while True:
        result = working_flow(step_add_input=step_add_input)
        result_image = result['image']
        cv2.imshow('result', result_image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
    print('Finish')
