import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import DataLoader

from model import Net

# 一些超參數設定
parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default=r"E:\Desktop\yolov5deepsort\Market-1501-v15.09.15\pytorch", type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true')
args = parser.parse_args()

# device
# 確認訓練設備
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
# 可以提升訓練速度，這個是在輸入圖像大小都保持不變時可以透過先找到最佳算法讓之後訓練時速度可以提升
# 在預設的情況下會是被關閉的，因為現在大部分的模型都會支持多尺度訓練這樣反而會降低效率
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
root = args.data_dir
# 資料集檔案位置
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")

# 訓練集的轉換方式
transform_train = torchvision.transforms.Compose([
    # 先將圖像進行四邊的填充後再做裁減，最終輸出的大小會是(128, 64)
    torchvision.transforms.RandomCrop((128, 64), padding=4),
    # 隨機水平翻轉
    torchvision.transforms.RandomHorizontalFlip(),
    # 轉換成tensor格式
    torchvision.transforms.ToTensor(),
    # 標準化圖像
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 驗證集的轉換方式
transform_test = torchvision.transforms.Compose([
    # 先將圖像進行四邊的填充後再做裁減，最終輸出的大小會是(128, 64)
    torchvision.transforms.Resize((128, 64)),
    # 轉換成tensor格式
    torchvision.transforms.ToTensor(),
    # 標準化圖像
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 透過ImageFolder構建dataset，同一種類別的照片會放在同一個資料夾底下，同時資料夾名稱就是類別名稱
# 構建訓練用的dataloader
trainloader = DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
    batch_size=64, shuffle=True
)
# 構建訓練用的dataloader，理論上來說shuffle在驗證時應該要關閉這裡不知道是不是有問題
testloader = DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
    batch_size=64, shuffle=True
)
# 獲取分類類別數，這裡的類別數量會是有多少種不同的人，每一個人表示一種類別
num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))
# 輸出類別數量
print("num_classes = %s" % num_classes)

# net definition
start_epoch = 0
# 實例化模型，並且將分類類別數傳入
net = Net(num_classes=num_classes)

# 訓練中斷時可以加載上次訓練到的進度接下去繼續訓練
if args.resume:
    # 這裡會直接指定預訓練的權重位置
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    # 加載預訓練模型
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    # 模型權重
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    # 最佳正確率
    best_acc = checkpoint['acc']
    # 到哪個epoch停下的
    start_epoch = checkpoint['epoch']

# 將模型轉換到訓練設備上
net.to(device)

# loss and optimizer
# 損失計算方式
criterion = torch.nn.CrossEntropyLoss()
# 優化器
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.


# train function for each epoch
def train(epoch):
    # 已看過
    # 開始訓練一個epoch
    print("\nEpoch : %d" % (epoch+1))
    # 將模型調整成訓練模式
    net.train()
    # 一些過程的數值保留
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    # 預設為20
    interval = args.interval
    # 紀錄開始時間
    start = time.time()
    # 遍歷一個epoch
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        # inputs shape [batch_size, 3, height, width]
        # labels shape [batch_size]
        inputs, labels = inputs.to(device), labels.to(device)
        # 正向傳播獲得預測結果
        outputs = net(inputs)
        # 進行損失計算
        loss = criterion(outputs, labels)

        # backward
        # 清空優化器
        optimizer.zero_grad()
        # 反向傳播
        loss.backward()
        optimizer.step()

        # accumurating
        # 計算損失值以及正確率
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        # 打印訓練情況
        if (idx+1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.
            start = time.time()
    
    return train_loss / len(trainloader), 1. - correct/total


def test(epoch):
    # 已看過
    global best_acc
    # 將模型調整到驗證模式
    net.eval()
    # 驗證的一些狀態紀錄
    test_loss = 0.
    correct = 0
    total = 0
    # 紀錄開始時間
    start = time.time()
    with torch.no_grad():
        # 遍歷一次驗證集
        for idx, (inputs, labels) in enumerate(testloader):
            # 就是單純驗證而已
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
            ))

    # saving checkpoint
    acc = 100.*correct / total
    # 保存最佳正確率
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss/len(testloader), 1. - correct/total


# plot figure
# 可視化訓練狀態需要的
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    # 已看過
    # 就是將訓練過程變成一張圖片進行可視化
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")


# lr decay
def lr_decay():
    # 已看過
    # 更新學習率
    global optimizer
    for params in optimizer.param_groups:
        # 將學習率調整成原先的十分之一
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    # 已看過
    # 設定要訓練的epoch數量
    total_epoches = 40
    for epoch in range(start_epoch, start_epoch+total_epoches):
        # 進行訓練
        train_loss, train_err = train(epoch)
        # 進行驗證
        test_loss, test_err = test(epoch)
        # 畫出訓練狀況圖
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        # 更新學習率
        if (epoch+1) % (total_epoches // 2) == 0:
            lr_decay()


if __name__ == '__main__':
    # 已看過
    main()
