import tensorrt
import torch
from torch import nn
from SpecialTopic.Deploy.OnnxToTensorRT.TensorrtBase import TensorrtBase

""" 
主要是確認在初始化模型時是否可以使用迴圈來構建網路結構
目前測試結果為可以
"""


def using_for_in_model_init():

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            conv_channels = [3, 64, 128, 256, 64]
            self.conv_layers = nn.ModuleList()
            for idx, conv_channel in enumerate(conv_channels[1:]):
                in_channel = conv_channels[idx]
                self.conv_layers.append(
                    nn.Conv2d(in_channel, conv_channel, kernel_size=3, padding=1, stride=1))
            self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
            self.fc_layers = nn.Linear(in_features=conv_channels[-1], out_features=10)

        def forward(self, x):
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            x = self.avg_pooling(x)
            x = self.flatten(x)
            out = self.fc_layers(x)
            return out

    model = Net()
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_names = ['images']
    output_names = ['preds']
    dynamic_axes = {'images': {0: 'batch_size'}, 'preds': {0: 'batch_size'}}
    images = torch.randn(2, 3, 32, 32)
    images = images.to(device)
    _ = model(images)
    onnx_file = 'TestTransfer.onnx'
    with torch.no_grad():
        torch.onnx.export(model, images, onnx_file, verbose=False, input_names=input_names, output_names=output_names,
                          opset_version=11, dynamic_axes=dynamic_axes)

    save_trt_engine_path = 'TestTransfer.trt'
    # trt_engine_path = 'TestTransfer.trt'
    trt_engine_path = None
    dynamic_shapes = {'images': ((1, 3, 32, 32), (2, 3, 32, 32), (3, 3, 32, 32))}
    tensor_engine = TensorrtBase(onnx_file_path=onnx_file, fp16_mode=True, trt_engine_path=trt_engine_path,
                                 save_trt_engine_path=save_trt_engine_path, dynamic_shapes=dynamic_shapes,
                                 max_batch_size=3)
    images = torch.randn((3, 3, 32, 32))
    input_datas = {'images': images.cpu().numpy()}
    output_shapes = ['preds']
    dynamic_shape = True
    tensorrt_preds = tensor_engine.inference(input_datas=input_datas, output_shapes=output_shapes,
                                             dynamic_shape=dynamic_shape)
    tensorrt_preds = tensorrt_preds[0]
    tensorrt_preds = tensorrt_preds.argmax(axis=1)
    print(tensorrt_preds)


if __name__ == '__main__':
    print('Testing torch -> onnx -> tensorrt')
    using_for_in_model_init()
