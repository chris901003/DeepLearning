# Pytorch To ONNX 踩坑紀錄

### 注意事項
不是每個模塊都可以順利轉成onnx結構，只有onnx有支援的映射關係的才可以轉換，或者是自行撰寫\
可以使用以下指令在不同構建onnx檔案時檢查當前是否在構建onnx過程
- torch.jit.is_tracing() = 如果是使用torch.jit.trace就會讓這個部分變成True
- torch.jit.is_scripting() = 如果是使用torch.jit.script就會讓這個部分變成True
透過上述指令可以讓某些模塊在構建onnx時跳過，例如計算損失函數或是正確率的函數在構建onnx檔案時就直接跳過，這樣可以避免無法轉換以及不必要的計算\
目前對於轉換成onnx還有許多不清楚的部分，到時在將模型轉換成onnx時有出現問題再到這裡進行記錄

### torch.onnx.export
如果直接使用torch.onnx.export的話就會根據傳入的資料進行追蹤，只有過程中有經果的模塊會被轉換成onnx圖資料其他的都不會\
同時直接使用torch.onnx.export會使用的模式為tracing模式，所以此模式比較適合沒有動態的模型，也就是在模型當中勁量不要使用for或是if的指令\
因為如果在進行tracing時沒有進入的部分就不會被轉成onnx的一部分\
接下來使用程式碼說明torch.onnx.export中的參數使用
```python
import torch
from torchvision import models
# 這裡使用script的方式來展示，因為我想我應該大部分都會使用script的方式來完成onnx製作
net = models.resnet34()
net = net.eval()
# 表示的為輸入的的資料，這裡給的名稱與模型當中的forward變數名稱不需要對應，只是順序需要相同就可以
inputs_name = ['inputs']
# 表示的為輸出的資料，同樣指示順序對應即可
outputs_name = ['outputs']
# 表示哪些資料的哪幾個維度是動態的，也就是不確定大小的，最常見的就會是batch_size的部分會是不確定的
# 這裡重點的部分只有index的部分，後面帶的名稱自己看得懂就行
dynamic_axes = {
    'inputs': {0: 'batch_size', 1: 'channel'},
    'preprocess_val': {0: 'batch_size'}
}
# 隨機給定一個資料，主要是要讓模型跑一次流程，讓torch可以記錄下模型圖結構
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    # 使用script方式記錄下來
    model_script = torch.jit.script(net)
    torch.onnx.export(model_script, x, 'net.onnx', input_names=inputs_name, output_names=outputs_name,
                      dynamic_axes=dynamic_axes)
```

### torch.jit.script
使用script的方式描述模型，在使用script時就會追蹤到for與if語句，這些判斷的部分都會被記錄下來，不過同時因為有記錄下動態語句，所以最後推理時\
的速度會相較於trace還要慢一點\
這裡要特別注意對於if的使用，目前找出的幾個規律，不一定正確如果有發現還有其他問題會進行改正
- 假設有兩種路可以走，必須保證兩條路出來的shape必須相同
- shape當中的值是否一定要相同目前沒有很確定，有些情況下可以不相同有些情況下又一定要一樣，可以直接看以下程式碼
```python
import torch
res = torch.randn(1, 3, 10, 10)
res = res.reshape(res.size(0), -1)
channel = res.size(1)
# 這樣是有問題的
if channel == 3:
    res = res[:10]
else:
    res = res[:11]

# 這樣是沒問題的
if channel == 3:
    mask = res > 0
    res = res[mask]
else:
    mask = res > 10
    res = res[mask]
```
- 如果在中間使用到例如卷積層結構，不會在意中間的shape是否相同，只要最後的輸出shape相同即可
```python
import torch
res = torch.randn(1, 3, 10, 10)
res = res.reshape(res.size(0), -1)
conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
conv3 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
channel = res.size(1)
if channel == 3:
    res = conv1(res)
    res = conv2(res)
else:
    res = conv3(res)
# 只要最終的shape相同即可
```
從torch.jit.script轉成onnx
```python
import torch
x = torch.randn(1, 3, 10, 10)
inputs_name, outputs_name, dynamic_axes = list(), list(), dict()
torch.onnx.export('model_script', x, 'net.onnx', input_names=inputs_name, output_names=outputs_name, 
                  dynamic_axes=dynamic_axes)
```
