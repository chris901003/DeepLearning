import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torchvision import utils
import torchvision


def get_cls_from_cfg(support, cfg):
    cls_type = cfg.pop('type', None)
    assert cls_type is not None, '在設定檔當中沒有指定type'
    assert cls_type in support, f'指定的{cls_type}尚未支援'
    cls = support[cls_type]
    return cls


class Compose:
    def __init__(self, pipeline_cfg):
        support_transform = {
            'LoadImageFromFile': LoadImageFromFile,
            'Resize': Resize,
            'Normalize': Normalize,
            'ToTensor': ToTensor,
            'Collect': Collect
        }
        self.transform = list()
        for pipeline in pipeline_cfg:
            transform_cls = get_cls_from_cfg(support_transform, pipeline)
            transform = transform_cls(**pipeline)
            self.transform.append(transform)

    def __call__(self, data):
        for transform in self.transform:
            data = transform(data)
        return data


class LoadImageFromFile:
    def __init__(self, key, img_type='RGB'):
        self.key = key
        self.img_type = img_type

    def __call__(self, data):
        assert os.path.isfile(data), '輸入的圖像路徑需要存在'
        img = cv2.imread(data)
        if self.img_type == 'RGB':
            assert len(img.shape) == 3 and img.shape[2] == 3, f'圖像{data}不是RGB格式'
        elif self.img_type == 'Gray':
            assert len(img.shape) == 2, f'圖像{data}不是灰度圖像'
            img = img[..., None]
        result = {
            self.key: img,
            'img_type': self.img_type
        }
        return result


class Resize:
    cv2_interp_codes = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }

    def __init__(self, keys, scale, interpolation, backend='cv2', keep_ratio=False):
        if isinstance(scale, int):
            scale = (scale, scale)
        self.keys = keys
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.backend = backend

    def __call__(self, data):
        for info_name in self.keys:
            img = data.get(info_name, None)
            assert img is not None, f'指定的{info_name}不在data當中'
            if self.backend == 'cv2':
                img = cv2.resize(img, self.scale, interpolation=self.cv2_interp_codes[self.interpolation])
            else:
                raise NotImplementedError(f'{self.backend}尚未支援')
            data[info_name] = img
        return data


class Normalize:
    def __init__(self, keys, std, mean):
        self.keys = keys
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)

    def __call__(self, data):
        for info_name in self.keys:
            img = data.get(info_name, None)
            assert img is not None, f'{info_name}不在data資料當中'
            img = (img - self.mean) / self.std
            data[info_name] = img
        return data


class ToTensor:
    def __init__(self, keys, reshaped=None):
        self.keys = keys
        self.reshaped = reshaped

    def __call__(self, data):
        for info_name in self.keys:
            info = data.get(info_name, None)
            assert info is not None, f'{info_name}不在data資料當中'
            if info_name in self.reshaped:
                info = cv2.cvtColor(info, cv2.COLOR_BGR2RGB)
                info = info.transpose(2, 0, 1)
            if isinstance(info, np.ndarray):
                info = torch.from_numpy(info)
            data[info_name] = info
        return data


class Collect:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        results = dict()
        for info_name in self.keys:
            info = data.get(info_name, None)
            assert info is not None, f'{info_name}不在data資料當中'
            results[info_name] = info
        return results


class GANDataset(Dataset):
    def __init__(self, data_path, dataset_cfg):
        assert os.path.isdir(data_path), '傳入的圖像路徑需要是資料夾'
        support_img_format = ['.jpg']
        self.imgs = list()
        for img_name in os.listdir(data_path):
            path = os.path.join(data_path, img_name)
            if os.path.isfile(path) and os.path.splitext(path)[1] in support_img_format:
                self.imgs.append(path)
        self.pipeline = Compose(dataset_cfg)

    def __getitem__(self, idx):
        data = self.imgs[idx]
        data = self.pipeline(data)
        return data

    def __len__(self):
        return len(self.imgs)


def CreateDataloader(cfg):
    dataloader = DataLoader(**cfg)
    return dataloader


def custom_collate_fn(batch):
    imgs = list()
    for info in batch:
        img = info.get('real_img', None)
        assert img is not None, '未獲取圖像資訊'
        imgs.append(img)
    imgs = torch.stack(imgs, dim=0)
    return imgs


def CreateModel(model_cfg, optimizer_cfg):
    support_model = {
        'StaticUnconditionalGAN': StaticUnconditionalGAN
    }
    model_cls = get_cls_from_cfg(support_model, model_cfg)
    model = model_cls(optimizer_cfg, **model_cfg)
    return model


def build_module(cfg):
    support_module = {
        'DCGANGenerator': DCGANGenerator,
        'DCGANDiscriminator': DCGANDiscriminator,
        'GANLoss': GANLoss
    }
    module_cls = get_cls_from_cfg(support_module, cfg)
    module = module_cls(**cfg)
    return module


def build_optimizer(model, cfg):
    support_optimizer = {
        'Adam': torch.optim.Adam
    }
    if isinstance(cfg, dict):
        result = dict()
        for key, value in cfg.items():
            optimizer_cls = get_cls_from_cfg(support_optimizer, value)
            module = getattr(model, key)
            optimizer = optimizer_cls(module.parameters(), **value)
            result[key] = optimizer
        return result
    else:
        optimizer_cls = get_cls_from_cfg(support_optimizer, cfg)
        optimizer = optimizer_cls(**cfg)
        return optimizer


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class StaticUnconditionalGAN(nn.Module):
    def __init__(self, optimizer_cfg, generator, discriminator, gan_loss):
        super(StaticUnconditionalGAN, self).__init__()
        self.generator = build_module(generator)
        if discriminator is not None:
            self.discriminator = build_module(discriminator)
        else:
            self.discriminator = None
        if gan_loss is not None:
            self.gan_loss = build_module(gan_loss)
        else:
            self.gan_loss = None
        self.optimizer = build_optimizer(self, optimizer_cfg)

    def get_disc_loss(self, outputs_dict):
        losses_dict = dict()
        losses_dict['loss_disc_fake'] = self.gan_loss(
            outputs_dict['disc_pred_fake'], target_is_real=False, is_disc=True)
        losses_dict['loss_disc_real'] = self.gan_loss(
            outputs_dict['disc_pred_real'], target_is_real=True, is_disc=True)
        loss = sum(_value for _key, _value in losses_dict.items())
        return loss

    def get_gen_loss(self, outputs_dict):
        losses_dict = dict()
        losses_dict['loss_disc_fake_g'] = self.gan_loss(
            outputs_dict['disc_pred_fake_g'], target_is_real=True, is_disc=False)
        loss = sum(_value for _key, _value in losses_dict.items())
        return loss

    def forward(self, real_imgs, train):
        batch_size = real_imgs.shape[0]
        set_requires_grad(self.discriminator, True)
        self.optimizer['discriminator'].zero_grad()
        with torch.no_grad():
            fake_imgs = self.generator(None, num_batch=batch_size)
        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            batch_size=batch_size)
        loss_disc = self.get_disc_loss(data_dict_)
        loss_disc.backward()
        self.optimizer['discriminator'].step()
        set_requires_grad(self.discriminator, False)
        self.optimizer['generator'].zero_grad()
        fake_imgs = self.generator(None, num_batch=batch_size)
        disc_pred_fake_g = self.discriminator(fake_imgs)
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake_g,
            batch_size=batch_size)
        loss_gen = self.get_gen_loss(data_dict_)
        loss_gen.backward()
        self.optimizer['generator'].step()
        return fake_imgs[:2], loss_gen


class DCGANGenerator(nn.Module):
    def __init__(self,
                 output_scale,
                 out_channels=3,
                 base_channels=1024,
                 input_scale=4,
                 noise_size=100):
        super(DCGANGenerator, self).__init__()
        self.output_scale = output_scale
        self.base_channels = base_channels
        self.input_scale = input_scale
        self.noise_size = noise_size
        self.num_upsample = int(np.log2(output_scale // input_scale))
        self.noise2feat = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise_size, out_channels=base_channels, kernel_size=4,
                               stride=1, padding=0),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True)
        )
        up_sampling = list()
        current_channels = base_channels
        for _ in range(self.num_upsample - 1):
            up_sampling.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=current_channels, out_channels=current_channels // 2,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(current_channels // 2), nn.ReLU(inplace=True)
            ))
            current_channels = current_channels // 2
        self.up_sampling = nn.Sequential(*up_sampling)
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=current_channels, out_channels=out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, num_batch=0, return_noise=False):
        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.noise_size
            if noise.ndim == 2:
                noise_batch = noise[:, :, None, None]
            elif noise.ndim == 4:
                noise_batch = noise
            else:
                raise ValueError('輸入的noise有誤')
        else:
            assert num_batch > 0
            noise_batch = torch.randn((num_batch, self.noise_size, 1, 1))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        noise_batch = noise_batch.to(device)
        x = self.noise2feat(noise_batch)
        x = self.up_sampling(x)
        x = self.output_layer(x)
        if return_noise:
            return dict(fack_img=x, noise_batch=noise_batch)
        return x


class DCGANDiscriminator(nn.Module):
    def __init__(self,
                 input_scale,
                 output_scale,
                 out_channels,
                 in_channels=3,
                 base_channels=128):
        super(DCGANDiscriminator, self).__init__()
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_downsample = int(np.log2(input_scale // output_scale))
        down_sampling = list()
        curr_channels = in_channels
        for i in range(self.num_downsample):
            in_channel = in_channels if i == 0 else base_channels * 2**(i - 1)
            out_channel = base_channels * 2**i
            if i == 0:
                down_sampling.append(nn.Sequential(
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(inplace=True)
                ))
            else:
                down_sampling.append(nn.Sequential(
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channel), nn.LeakyReLU(inplace=True)
                ))
            curr_channels = base_channels * 2 ** i
        self.down_sampling = nn.Sequential(*down_sampling)
        self.output_layer = nn.Conv2d(in_channels=curr_channels, out_channels=out_channels,
                                      kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        n = x.shape[0]
        x = self.down_sampling(x)
        x = self.output_layer(x)
        return x.view(n, -1)


class GANLoss(nn.Module):
    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f'{gan_type}尚未提供')

    def get_target_label(self, input, target_is_real):
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss if is_disc else loss * self.loss_weight


def run(model, device, train_epoch, train_dataloader, val_epoch=None, val_dataloader=None):
    if val_epoch is not None:
        assert val_dataloader is not None, '須提供驗證的dataloader'
    best_loss = 10000
    for epoch in range(1, train_epoch + 1):
        total_gen_loss = train_one_epoch(model, device, train_dataloader, best_loss, epoch)
        best_loss = min(total_gen_loss, best_loss)
        if val_epoch is not None:
            if epoch % val_epoch == 0:
                val_one_epoch(model, device, val_dataloader, epoch)
    print('Finish training')


def train_one_epoch(model, device, dataloader, best_loss, epoch):
    total_gen_loss = 0
    total_picture = 0
    model.train()
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch}: ',
              postfix=f'Gen loss {total_gen_loss}', mininterval=1) as pbar:
        for imgs in dataloader:
            total_picture += len(imgs)
            imgs = imgs.to(device)
            fake_img, gen_loss = model(imgs, train=True)
            fake_img = (fake_img[:, [2, 1, 0]] + 1.) / 2.
            save_path = f'/Users/huanghongyan/Documents/DeepLearning/MMSegmentation_work_dir/test{epoch}.jpg'
            utils.save_image(fake_img, save_path)
            total_gen_loss += gen_loss
            pbar.set_postfix_str(f'Gen loss {total_gen_loss}')
            pbar.update(1)
    if total_gen_loss < best_loss:
        torch.save(model.state_dict(), 'best_model.pkt')
    return total_gen_loss


def val_one_epoch(model, device, dataloader, epoch):
    pass


def main():
    data_path = '/Users/huanghongyan/Documents/DeepLearning/mmgeneration/data/ffhq'
    dataset_cfg = [
        {'type': 'LoadImageFromFile', 'key': 'real_img', 'img_type': 'RGB'},
        {'type': 'Resize', 'keys': ['real_img'], 'scale': (64, 64), 'keep_ratio': False,
         'interpolation': 'bilinear', 'backend': 'cv2'},
        {'type': 'Normalize', 'keys': ['real_img'], 'std': [127.5] * 3, 'mean': [127.5] * 3},
        {'type': 'ToTensor', 'keys': ['real_img'], 'reshaped': ['real_img']},
        {'type': 'Collect', 'keys': ['real_img']}
    ]
    gan_dataset = GANDataset(data_path, dataset_cfg)
    dataloader_cfg = {
        'dataset': gan_dataset,
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 1,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': custom_collate_fn
    }
    gan_dataloader = CreateDataloader(dataloader_cfg)
    model_cfg = {
        'type': 'StaticUnconditionalGAN',
        'generator': {
            'type': 'DCGANGenerator',
            'output_scale': 64,
            'base_channels': 1024
        },
        'discriminator': {
            'type': 'DCGANDiscriminator',
            'input_scale': 64,
            'output_scale': 4,
            'out_channels': 1
        },
        'gan_loss': {
            'type': 'GANLoss',
            'gan_type': 'vanilla'
        }
    }
    optimizer_cfg = {
        'generator': {
            'type': 'Adam',
            'lr': 0.0002,
            'betas': (0.5, 0.999)
        },
        'discriminator': {
            'type': 'Adam',
            'lr': 0.0002,
            'betas': (0.5, 0.999)
        }
    }
    dcgan_model = CreateModel(model_cfg, optimizer_cfg)
    pretrained = 'best_model.pkt'
    dcgan_model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_epoch = 100
    run(dcgan_model, device, train_epoch, gan_dataloader)


if __name__ == '__main__':
    main()
    print('Finish')
