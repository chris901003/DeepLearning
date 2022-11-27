import argparse
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
from PIL import Image
from tqdm import tqdm
import os


def args_parse():
    parser = argparse.ArgumentParser()
    # 需要提供系統字體，這裡需要每個數字的樣子才可以進行辨識
    # 資料格式，圖像中的數字與檔名相同，並且需要是[.jpg, .JPG, .jpeg, .JPEG]其中一種
    parser.add_argument('--system-number-set', type=str, default='SystemNumberSet')
    # 影片路徑
    parser.add_argument('--video-path', type=str, default='SystemNumberSet/rgb.mp4')
    # 保存文字檔位置
    parser.add_argument('--save-path', type=str, default='remain.txt')
    # 將框框位置填入，這裡請由左至右並且格式為[[xmin, ymin, xmax, ymax]]
    parser.add_argument('--boxes-place', type=int, default=[[10, 20, 100, 120], [120, 20, 220, 120]], nargs='+')
    # 如果要看框選位置就改成True，此時就只會看一下框選位置
    parser.add_argument('--view-box', type=bool, default=False)
    args = parser.parse_args()
    return args


class SystemNumberDataset(Dataset):
    def __init__(self, data_path):
        support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
        self.images_path = [os.path.join(data_path, image_name) for image_name in os.listdir(data_path)
                            if os.path.splitext(image_name)[1] in support_image_format]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        label = int(os.path.splitext(os.path.basename(image_path))[0])
        label = torch.tensor(label)
        result = dict(image=image, label=label)
        return result

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images = list()
        labels = list()
        for info in batch:
            images.append(info['image'])
            labels.append(info['label'])
        images = torch.stack(images)
        labels = torch.stack(labels)
        return images, labels


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # out shape = 64 x 32 x 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        # out shape = 128 x 16 x 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        # out shape = 64 x 8 x 8
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        # out shape = 64 x 1 x 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # out shape = 10
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def prepare_recognition_model(data_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = SystemNumberDataset(data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=train_dataset.collate_fn)
    model = Net()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    print('Start training system number')
    for epoch in tqdm(range(1, 101)):
        acc = 0
        tot = 0
        for image, label in train_dataloader:
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)
            out = model(image)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            pred = out.argmax(dim=-1)
            acc += pred.eq(label).sum().item()
            tot += image.size(0)
        if epoch == 100:
            print(acc / tot)
    torch.save(model.state_dict(), 'SystemNumberWeight.pth')
    print('Finish pretrain system number')


def view_box_place(video_path, boxes_place):
    cap = cv2.VideoCapture(video_path)
    _, _ = cap.read()
    ret, image = cap.read()
    assert ret, '給定影片有問題，無法進行讀取'
    image_height, image_width = image.shape[:2]
    for idx, box_place in enumerate(boxes_place):
        xmin, ymin, xmax, ymax = box_place
        if xmin < 0 or ymin < 0 or xmax >= image_width or ymax >= image_height:
            print(f'Box index {idx} out of image')
            print(f'Image max width: {image_width}. Image max height: {image_height}')
            continue
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.show()


def preprocess_picture(pictures):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    images = list()
    for picture in pictures:
        image = Image.fromarray(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
        image = transform(image)
        images.append(image)
    images = torch.stack(images)
    return images


def create_remain_with_weight(video_path, save_path, boxes_place):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Net()
    # model.load_state_dict(torch.load('SystemNumberWeight.pth', map_location='cpu'))
    model = model.to(device)
    model.eval()
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    weight_record = list()
    while True:
        ret, image = cap.read()
        if ret:
            image_height, image_width = image.shape[:2]
            numbers_picture = list()
            for idx, box_place in enumerate(boxes_place):
                xmin, ymin, xmax, ymax = box_place
                if xmin < 0 or ymin < 0 or xmax >= image_width or ymax >= image_height:
                    raise RuntimeError('標註範圍大於影片大小，請使用view模式檢查標註框')
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                picture = image[ymin:ymax, xmin:xmax, :]
                numbers_picture.append(picture)
            number_pictures = preprocess_picture(numbers_picture)
            with torch.no_grad():
                number_pictures = number_pictures.to(device)
                predicts = model(number_pictures)
            predicts = predicts.argmax(dim=-1)
            print('f')

            cv2.imshow('img', image)
        else:
            break
        if cv2.waitKey(1) == ord('q'):
            break


def main():
    args = args_parse()
    system_number_set_path = args.system_number_set
    video_path = args.video_path
    save_path = args.save_path
    boxes_place = args.boxes_place
    view_box = args.view_box
    assert system_number_set_path is not None and os.path.exists(system_number_set_path), '需要提供系統的字體'
    assert video_path is not None and os.path.exists(video_path), '需要提供影片路徑'
    assert save_path is not None, '需要指定保存結果路徑'
    if view_box:
        view_box_place(video_path, boxes_place)
    else:
        # prepare_recognition_model(system_number_set_path)
        create_remain_with_weight(video_path, save_path, boxes_place)
    if os.path.exists('SystemNumberWeight.pth'):
        os.remove('SystemNumberWeight.pth')


if __name__ == '__main__':
    main()
