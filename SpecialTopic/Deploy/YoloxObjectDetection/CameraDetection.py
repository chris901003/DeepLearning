import argparse
import cv2
import time
from api import create_tensorrt_engine, tensorrt_engine_detect_image
from SpecialTopic.ST.utils import get_classes


def parse_args():
    parser = argparse.ArgumentParser()
    # 使用的相機ID
    parser.add_argument('--camera-id', type=int, default=0)
    # FPS值
    parser.add_argument('-fps', type=int, default=30)
    # 分類類別文件
    parser.add_argument('--classes-file', type=str, default=r'C:\Dataset\FoodDetectionDataset\classes.txt')
    # Onnx檔案位置
    parser.add_argument('--onnx-file', type=str, default='YoloxObjectDetectionL_Simplify.onnx')
    # TensorRT序列化保存位置，如果有提供就可以直接使用已經序列化好的引擎，可以大幅度減少構建時間
    parser.add_argument('--trt-engine-path', type=str, default='YoloxObjectDetectionL_Simplify.trt')
    # 將本次的引擎序列化後保存下來
    parser.add_argument('--save-trt-engine-path', type=str, default=None)
    # 是否使用fp16模式，使用後可以提升速度但是會減少精度
    parser.add_argument('--fp16', action='store_false')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tensorrt_engine = create_tensorrt_engine(onnx_file_path=args.onnx_file, fp16_mode=args.fp16,
                                             trt_engine_path=args.trt_engine_path,
                                             save_trt_engine_path=args.save_trt_engine_path,
                                             trt_logger_level='VERBOSE')
    classes_name, num_classes = get_classes(args.classes_file)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    while True:
        ret, img = cap.read()
        if ret:
            sTime = time.time()
            results = tensorrt_engine_detect_image(tensorrt_engine=tensorrt_engine, image=img, num_classes=num_classes)
            eTime = time.time()
            fps = 1 / (eTime - sTime)
            labels, scores, boxes = results
            for label, score, box in zip(labels, scores, boxes):
                ymin, xmin, ymax, xmax = box
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                info = str(classes_name[label]) + ' || ' + str(score)
                cv2.putText(img, info, (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (89, 214, 210), 2, cv2.LINE_AA)
            cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
