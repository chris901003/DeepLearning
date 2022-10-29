import onnxruntime
import onnx
import torch
import cv2
import numpy as np


def main():
    onnx_model = onnx.load('net.onnx')
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    x = torch.randn(1, 3, 224, 224)
    preprocess_val = torch.randn(1, 64, 10, 10)
    ort_session = onnxruntime.InferenceSession('net.onnx')
    ort_inputs = {'inputs': x.numpy(), 'preprocess_val': preprocess_val.numpy()}
    ort_outputs = ort_session.run(['out', 'outputs'], ort_inputs)[0]
    print(ort_outputs.shape)


def resnet_onnx():
    from torchvision import transforms
    from torchvision import models
    from PIL import Image
    import time
    transform_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open('/Users/huanghongyan/Downloads/test.jpeg')
    image = transform_data(image)
    input_image = torch.stack([image for _ in range(12)])
    ort_inputs = {'images_input': input_image.numpy()}
    ort_session = onnxruntime.InferenceSession('resnet.onnx')
    ort_outputs = ort_session.run(['preds_output'], ort_inputs)[0]
    pred = ort_outputs.argmax(axis=1)
    print(pred)

    onnx_time_record = list()
    for _ in range(10):
        start_time = time.time()
        _ = ort_session.run(['preds_output'], ort_inputs)[0]
        end_time = time.time()
        onnx_time_record.append(end_time - start_time)
    print(f'Onnx average running time: {sum(onnx_time_record) / len(onnx_time_record)}')

    resnet = models.resnet18(pretrained=True)
    resnet = resnet.eval()
    pred = resnet(input_image).detach().numpy()
    pred = pred.argmax(axis=1)
    print(pred)
    pytorch_time_record = list()
    for _ in range(10):
        start_time = time.time()
        with torch.no_grad():
            _ = resnet(input_image)
        end_time = time.time()
        pytorch_time_record.append(end_time - start_time)
    print(f'Pytorch average running time: {sum(pytorch_time_record) / len(pytorch_time_record)}')


if __name__ == '__main__':
    print('Test using onnx')
    resnet_onnx()
