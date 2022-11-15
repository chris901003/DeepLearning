import argparse
import cv2
import time
from SpecialTopic.Deploy.SegmentationNet.api import create_tensorrt_engine, tensorrt_engine_detect_image


def parse_args():
    parser = argparse.ArgumentParser()
    # 使用的相機ID
    parser.add_argument('--camera-id', type=int, default=0)
    # 相機FPS值
    parser.add_argument('--fps', type=int, default=30)
    # onnx檔案保存路徑
    parser.add_argument('--onnx-file', type=str, default=None)
    # tensorrt引擎保存路徑
    parser.add_argument('--trt-engine-path', type=str, default='SegmentationNetNano.trt')
    # 保存tensorrt引擎序列化後的資料
    parser.add_argument('--save-engine-path', type=str, default='SegmentationNetNano.trt')
    # 是否要啟用fp16模式進行推理
    parser.add_argument('--fp16', action='store_false')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tensorrt_engine = create_tensorrt_engine(onnx_file_path=args.onnx_file, fp16_mode=args.fp16,
                                             trt_engine_path=args.trt_engine_path,
                                             save_trt_engine_path=args.save_engine_path)
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    while True:
        ret, img = cap.read()
        if ret:
            sTime = time.time()
            draw_image_mix, draw_image, seg_pred = tensorrt_engine_detect_image(tensorrt_engine, image_info=img)
            eTime = time.time()
            fps = 1 / (eTime - sTime)
            cv2.putText(draw_image_mix, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('img', draw_image_mix)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
