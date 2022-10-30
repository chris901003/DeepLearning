import time
import onnx
import torch
import torchvision
import onnxruntime


model = torchvision.models.resnet50(pretrained=True).cuda()
input_names = ['input']
output_names = ['output']
image = torch.randn(1, 3, 224, 224).cuda()
onnx_file = './resnet50.onnx'
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
torch.onnx.export(model, image, onnx_file, verbose=False, input_names=input_names, output_names=output_names,
                  opset_version=11, dynamic_axes=dynamic_axes)

net = onnx.load('./resnet50.onnx')
onnx.checker.check_model(net)

model.eval()
with torch.no_grad():
    output1 = model(image)

image = torch.randn(4, 3, 224, 224).cuda()
session = onnxruntime.InferenceSession('./resnet50.onnx', providers=['CUDAExecutionProvider'])
session.get_modelmeta()
output2 = session.run(['output'], {'input': image.cpu().numpy()})
print(f'{output1.mean()}vs{output2[0].mean()}')
