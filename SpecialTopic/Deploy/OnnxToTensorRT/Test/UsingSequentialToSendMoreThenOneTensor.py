import tensorrt
import torch
from torch import nn
from SpecialTopic.Deploy.OnnxToTensorRT.TensorrtBase import TensorrtBase


def using_sequential_to_send_more_then_one_tensor():
    # 目前無法檢測出問題，怎麼改都可以轉成tensorrt

    class ExpandModel(nn.Module):
        def __init__(self):
            super(ExpandModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
            self.conv3 = nn.Conv2d(3, 64, kernel_size=7, padding=3)

        def forward(self, x):
            output = []
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            x3 = self.conv3(x)
            output.append(x1)
            output.append(x2)
            output.append(x3)
            return output

    class LastModel(nn.Module):
        def __init__(self):
            super(LastModel, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_features=64 * 3, out_features=10)

        def forward(self, x):
            x1, x2, x3 = x
            x1 = self.avg_pool(x1)
            x2 = self.avg_pool(x2)
            x3 = self.avg_pool(x3)
            x = torch.cat((x1, x2, x3), dim=1)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)
            return x

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.expand_model = ExpandModel()
            self.last_model = LastModel()
            self.seq = nn.Sequential(self.expand_model, self.last_model)

        def forward(self, x):
            out = self.seq(x)
            return out
            # out = self.expand_model(x)
            # out = self.last_model(out)
            # return out

    model = Net()
    model.eval()
    input_names = ['images']
    output_names = ['preds']
    images = torch.randn(1, 3, 224, 224)
    _ = model(images)
    onnx_file = 'test.onnx'
    with torch.no_grad():
        torch.onnx.export(model, images, onnx_file, verbose=True, input_names=input_names, output_names=output_names,
                          opset_version=11)
    save_trt_engine_path = 'test.trt'
    # trt_engine_path = 'test.trt'
    trt_engine_path = None
    tensor_engine = TensorrtBase(onnx_file_path=onnx_file, fp16_mode=True, trt_engine_path=trt_engine_path,
                                 save_trt_engine_path=save_trt_engine_path, max_batch_size=1)
    input_datas = {'images': images.cpu().numpy()}
    output_shapes = ['preds']
    dynamic_shape = False
    tensorrt_preds = tensor_engine.inference(input_datas=input_datas, output_shapes=output_shapes,
                                             dynamic_shape=dynamic_shape)
    tensorrt_preds = tensorrt_preds[0]
    print(tensorrt_preds.shape)


if __name__ == '__main__':
    using_sequential_to_send_more_then_one_tensor()
