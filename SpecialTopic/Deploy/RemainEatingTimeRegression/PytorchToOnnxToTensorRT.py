import tensorrt
import argparse
from SpecialTopic.Deploy.RemainEatingTimeRegression.api import create_onnx_file, create_tensorrt_engine


def parse_args():
    parser = argparse.ArgumentParser('從Pytorch到Onnx到簡化Onnx到TensorRT')
    parser.add_argument('--pretrained-path', type=str, default='./prepare/regression_0.pth')
    parser.add_argument('--setting-path', type=str, default='./prepare/setting_0.json')
    parser.add_argument('--onnx-file-name', type=str, default='RemainEatingTimeRegression.onnx')
    parser.add_argument('--onnx-simplify-file-name', type=str, default='RemainEatingTimeRegression_Simplify.onnx')
    parser.add_argument('--trt-file-name', type=str, default='RemainEatingTimeRegression.trt')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    create_onnx_file(setting=args.setting_path, pretrained=args.pretrained_path, onnx_file_name=args.onnx_file_name,
                     simplify_file_name=args.onnx_simplify_file_name)
    create_tensorrt_engine(onnx_file_path=args.onnx_simplify_file_name, save_trt_engine_path=args.trt_file_name)
    print('Finish')


if __name__ == '__main__':
    main()
