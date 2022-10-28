import onnxruntime as ort
import onnx
import torch
import numpy as np


def main():
    onnx_model = onnx.load('test.onnx')
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))
    ort_sess = ort.InferenceSession('test.onnx')
    x = torch.randn(1, 3, 28, 28)
    y = torch.tensor([456])
    outputs = ort_sess.run(None, {'x': x.numpy(), 'y': y.numpy()})
    print(outputs)


if __name__ == '__main__':
    print('Test using onnx')
    main()
