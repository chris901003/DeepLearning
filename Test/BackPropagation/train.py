import argparse
import os
from torch.utils.data import DataLoader
from model import FullyConnectionModel
from dataset import MnistDataset
from fit_utils import fit_one_epoch_with_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Epoch', type=int, default=10)
    parser.add_argument('--lr', type=int, default=1e-1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--train-annotation-path', type=str, default='./train_annotation.txt')
    parser.add_argument('--eval-annotation-path', type=str, default='./eval_annotation.txt')
    parser.add_argument('--input-size', type=int, default=[28, 28], nargs='+')
    parser.add_argument('--train-data-prefix', type=str, default='')
    parser.add_argument('--eval-data-prefix', type=str, default='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_size = args.input_size
    if isinstance(input_size, int):
        input_size = [input_size, input_size]
    num_classes = args.num_classes
    model_cfg = [512, 256, 64, num_classes]
    model = FullyConnectionModel(model_cfg, input_size[0] * input_size[1])
    if not os.path.exists(args.eval_annotation_path):
        print('Using training annotation to eval annotation')
        args.eval_annotation_path = args.train_annotation_path
    batch_size = args.batch_size
    train_dataset = MnistDataset(args.train_annotation_path, args.train_data_prefix)
    train_dataloader_cfg = {
        'dataset': train_dataset,
        'batch_size': batch_size,
        'drop_last': False,
        'shuffle': True,
        'num_workers': 1,
        'pin_memory': True,
        'collate_fn': train_dataset.collate_fn
    }
    train_dataloader = DataLoader(**train_dataloader_cfg)
    test_dataset = MnistDataset(args.eval_annotation_path, args.eval_data_prefix)
    test_dataloader_cfg = {
        'dataset': train_dataset,
        'batch_size': batch_size,
        'drop_last': False,
        'shuffle': True,
        'num_workers': 1,
        'pin_memory': True,
        'collate_fn': test_dataset.collate_fn
    }
    test_dataloader = DataLoader(**test_dataloader_cfg)
    for epoch in range(1, args.Epoch + 1):
        fit_one_epoch_with_batch(model, epoch, args.Epoch, train_dataloader, test_dataloader, args.lr, num_classes)
        args.lr = args.lr * 0.1


if __name__ == '__main__':
    main()
    print('Finish')
