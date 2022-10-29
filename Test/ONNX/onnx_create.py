import torch
from torch import nn
import onnxruntime
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = [3, 64, 512, 256, 128, 64]
        self.cls = 10
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2dd = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.seq1 = nn.Sequential(self.conv2d, self.bn1)
        self.seq2 = nn.Sequential(self.conv2dd, self.bn2)
        self.convs = nn.ModuleList()
        for idx in range(1, len(self.conv_layers)):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=self.conv_layers[idx - 1], out_channels=self.conv_layers[idx], kernel_size=3,
                          padding=1),
                nn.BatchNorm2d(num_features=self.conv_layers[idx]),
                nn.ReLU(inplace=True)
            )
            self.convs.append(layer)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=self.conv_layers[-1], out_features=self.cls)

    def step1(self, y, t):
        res = y + t.item()
        return res

    def forward(self, x, y):
        if not torch.jit.is_tracing():
            if isinstance(y, int):
                y = torch.tensor(y)
        channel = y.shape[1]
        if channel == 3:
            res = self.conv2d(y)
            res = self.conv_mid(res)
        else:
            res = self.conv2dd(y)
        return x, res


def main():
    net = Net()
    net.eval()
    x = torch.randn(1, 3, 224, 224)
    y = torch.randn(1, 3, 10, 10)
    output = net(x, y)
    inputs_name = ['inputs', 'preprocess_val']
    outputs_name = ['outputs', 'out']
    dynamic_axes = {
        'inputs': {0: 'batch_size'},
        'preprocess_val': {1: 'channel'}
    }
    with torch.no_grad():
        model_script = torch.jit.script(net)
        torch.onnx.export(model_script, (x, y), 'net.onnx', input_names=inputs_name, output_names=outputs_name,
                          dynamic_axes=dynamic_axes)
    ort_session = onnxruntime.InferenceSession('net.onnx')


def resnet_onnx_create():
    from torchvision import transforms
    from torchvision import models
    from PIL import Image
    transform_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open('/Users/huanghongyan/Downloads/test.jpeg')
    image = transform_data(image)
    input_image = torch.stack([image for _ in range(2)])

    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    # input_image = torch.randn(3, 3, 224, 224)
    output = resnet(input_image)
    print(output.argmax(dim=1))
    inputs_name = ['images_input']
    outputs_name = ['preds_output']
    dynamic_axes = {
        'images_input': {0: 'batch_size'},
        'preds_output': {0: 'batch'}
    }
    with torch.no_grad():
        model_script = torch.jit.script(resnet)
        torch.onnx.export(model_script, input_image, 'resnet.onnx', input_names=inputs_name, output_names=outputs_name,
                          dynamic_axes=dynamic_axes)
        torch.onnx.export(resnet, input_image, 'resnet_forward.onnx', input_names=inputs_name,
                          output_names=outputs_name, dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    print('Testing onnx create')
    resnet_onnx_create()
