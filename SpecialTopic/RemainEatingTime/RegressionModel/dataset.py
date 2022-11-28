import pickle
import torch
from torch.utils.data.dataset import Dataset


class RegressionDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_list = self.parse_dataset(dataset_path)

    def __getitem__(self, idx):
        data = self.dataset_list[idx]
        return data

    def __len__(self):
        return len(self.dataset_list)

    @staticmethod
    def parse_dataset(dataset_path):
        with open(dataset_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def collate_fn(batch):
        remain_list, remain_time_list = list(), list()
        for info in batch:
            remain = info.get('remain', None)
            remain_time = info.get('remain_time', None)
            assert remain is not None, '缺少remain資料'
            assert remain_time is not None, '缺少remain time資料'
            remain_list.append(remain)
            remain_time_list.append(remain_time)
        remain_list = torch.Tensor(remain_list).to(torch.long)
        remain_time_list = torch.Tensor(remain_time_list).to(torch.long)
        return remain_list, remain_time_list
