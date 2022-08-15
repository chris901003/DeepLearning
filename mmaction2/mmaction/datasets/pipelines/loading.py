# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import io
import os
import os.path as osp
import shutil
import warnings

import mmcv
import numpy as np
import torch
from mmcv.fileio import FileClient
from torch.nn.modules.utils import _pair

from ...utils import get_random_string, get_shm_dir, get_thread_id
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadHVULabel:
    """Convert the HVU label from dictionaries to torch tensors.

    Required keys are "label", "categories", "category_nums", added or modified
    keys are "label", "mask" and "category_mask".
    """

    def __init__(self, **kwargs):
        self.hvu_initialized = False
        self.kwargs = kwargs

    def init_hvu_info(self, categories, category_nums):
        assert len(categories) == len(category_nums)
        self.categories = categories
        self.category_nums = category_nums
        self.num_categories = len(self.categories)
        self.num_tags = sum(self.category_nums)
        self.category2num = dict(zip(categories, category_nums))
        self.start_idx = [0]
        for i in range(self.num_categories - 1):
            self.start_idx.append(self.start_idx[-1] + self.category_nums[i])
        self.category2startidx = dict(zip(categories, self.start_idx))
        self.hvu_initialized = True

    def __call__(self, results):
        """Convert the label dictionary to 3 tensors: "label", "mask" and
        "category_mask".

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        if not self.hvu_initialized:
            self.init_hvu_info(results['categories'], results['category_nums'])

        onehot = torch.zeros(self.num_tags)
        onehot_mask = torch.zeros(self.num_tags)
        category_mask = torch.zeros(self.num_categories)

        for category, tags in results['label'].items():
            # skip if not training on this category
            if category not in self.categories:
                continue
            category_mask[self.categories.index(category)] = 1.
            start_idx = self.category2startidx[category]
            category_num = self.category2num[category]
            tags = [idx + start_idx for idx in tags]
            onehot[tags] = 1.
            onehot_mask[start_idx:category_num + start_idx] = 1.

        results['label'] = onehot
        results['mask'] = onehot_mask
        results['category_mask'] = category_mask
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'hvu_initialized={self.hvu_initialized})')
        return repr_str


@PIPELINES.register_module()
class SampleFrames:
    """Sample frames from the video.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDataset``, ``RawframeDataset``,
            etc), see this: https://github.com/open-mmlab/mmaction2/pull/89.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 keep_tail_frames=False):
        """ 已看過，對影片進行幀採樣
        Args:
            clip_len: 每個採樣輸出剪輯的幀數量
            frame_interval: 相鄰採樣點的時間間隔
            num_clips: 需要分成幾個剪輯數
            temporal_jitter: 是否啟用時間抖動
            twice_sample: 當測試時是否啟用二次採樣
            out_of_bound_opt: 當需要的幀數已經超過片長的處理方式，可以選擇重複最後一幀或是將影片是為環狀
            test_mode: 是否為測試模式
            start_index: 開始幀的index
            keep_tail_frames: 採樣時是否保留最後一幀
        """

        # 將傳入的參數進行保存
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        # 檢查out_of_bound_opt需要是loop或是repeat_last
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            # 如果有設定start_index會跳出警告，這裡已經將start_index的功能改到dataset當中
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        # 已看過，獲取訓練階段的剪輯幀的偏移量
        # 這裡會計算出平均經過多少幀會提取一次，並且決定好位置後還會有小幅度的隨機偏移，這個幅度會在[0, avg_interval]之間
        # 如果傳入的影片總幀數不到最少的幀數就會用0 indices代替

        # 計算出ori_clip_len，這裡會是總共需要的幀數乘上每幀之間的間隔，就可以獲取原先需要多少幀的影片
        ori_clip_len = self.clip_len * self.frame_interval

        if self.keep_tail_frames:
            # 如果有設定keep_tail_frames就會到這裡
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        else:
            # 如果沒有設定keep_tail_frames就會到這裡
            # 計算出平均的距離，這裡會是用(總幀數 - 每個片段需要的幀數 + 1) // 總共需要的片段量
            # 這裡會需要剪掉ori_clip_len的原因是因為之後我們會做base_offsets加上一個隨機值範圍在[0, avg_interval]
            # 如果我們有先預留一個ori_clip_len就不會超出最後的長度
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                # 如果計算出來的avg_interval大於0就會到這裡
                # 計算基底的偏移量，這裡計算出的是每個片段的開頭位置
                base_offsets = np.arange(self.num_clips) * avg_interval
                # 獲取clip_offsets，透過基底的偏移量再加上隨機生成的數字，這裡的數字會是從[0, avg_interval]
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                # 如果總共的幀數比max(需要的片段數量, 一個片段需要的長度)要多就會到這裡
                # 隨機從[0, 總幀數 - 一個片段需要的長度 + 1]取num_clips個，最後進行排序就獲得clip_pffsets
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                # 如果avg_interval剛好是0就會這裡
                # 用比例進行分配
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                # 這裡就會是依據比例進行分配clip_offsets的位置
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                # 其他情況就全部都給0，全部的片段都從0幀開始取
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        # 最後回傳clip_offsets
        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        # 已看過，根據當前的模式獲取剪輯幀的偏移量
        if self.test_mode:
            # 如果是測試模式會到這裡
            clip_offsets = self._get_test_clips(num_frames)
        else:
            # 如過是訓練模式就會到這裡
            clip_offsets = self._get_train_clips(num_frames)

        # 最終返回結果
        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # 已看過，執行樣本幀處理

        # 獲取當前影片總共有多少幀
        total_frames = results['total_frames']

        # 使用_sample_clips計算clip_offsets，clip_offsets = list[int]，list長度就是要切的片段數量
        clip_offsets = self._sample_clips(total_frames)
        # frame_inds = ndarray shape [num_clips, clip_len]，這個資料就是我們要取哪幾幀的圖像
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        # 透過concatenate將全部的值拼接起來，frame_inds = ndarray shape [num_clips * clip_len]
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            # 如果有設定temporal_jitter就會到這裡
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        # 將frame_inds進行通道調整 [num_clips * clip_len] -> [num_clips, clip_len]
        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            # 如果設定超出影片總幀數的地方要用loop方式處理就會到這裡
            # 這裡簡單使用mod的方式將超出的部分變回影片開頭的幀數
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            # 如果設定超出影片總幀數的地方要用repeat_last處理就會這裡
            # 獲取哪些部分的index是合法的
            safe_inds = frame_inds < total_frames
            # 用safe去推出哪些部分是超出影片幀數的
            unsafe_inds = 1 - safe_inds
            # 找到合法當中最大的index值
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            # 將合法的地方提取出來，不合法的index用最大的合法index代替
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            # 更新frame_inds資料
            frame_inds = new_inds
        else:
            # 其他設定就會直接報錯，這裡只接受兩種方式
            raise ValueError('Illegal out_of_bound option.')

        # 獲取start_index的資訊
        start_index = results['start_index']
        # 將frame_inds進行調整並且將所有的index平移start_index，frame_inds shape [num_clip * clip_len]
        frame_inds = np.concatenate(frame_inds) + start_index
        # 將需要的幀數保存
        results['frame_inds'] = frame_inds.astype(np.int)
        # 將每個片段需要的幀數保存
        results['clip_len'] = self.clip_len
        # 將每幀之間的間隔保存
        results['frame_interval'] = self.frame_interval
        # 將需要的片段數保存
        results['num_clips'] = self.num_clips
        # 最後回傳更新後的results
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class UntrimmedSampleFrames:
    """Sample frames from the untrimmed video.

    Required keys are "filename", "total_frames", added or modified keys are
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): The length of sampled clips. Default: 1.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 16.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDataset``, ``RawframeDataset``,
            etc), see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self, clip_len=1, frame_interval=16, start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        start_index = results['start_index']

        clip_centers = np.arange(self.frame_interval // 2, total_frames,
                                 self.frame_interval)
        num_clips = clip_centers.shape[0]
        frame_inds = clip_centers[:, None] + np.arange(
            -(self.clip_len // 2), self.clip_len -
            (self.clip_len // 2))[None, :]
        # clip frame_inds to legal range
        frame_inds = np.clip(frame_inds, 0, total_frames - 1)

        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval})')
        return repr_str


@PIPELINES.register_module()
class DenseSampleFrames(SampleFrames):
    """Select frames from the video by dense sample strategy.

    Required keys are "filename", added or modified keys are "total_frames",
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        sample_range (int): Total sample range for dense sample.
            Default: 64.
        num_sample_positions (int): Number of sample start positions, Which is
            only used in test mode. Default: 10. That is to say, by default,
            there are at least 10 clips for one input sample in test mode.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 *args,
                 sample_range=64,
                 num_sample_positions=10,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_range = sample_range
        self.num_sample_positions = num_sample_positions

    def _get_train_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in train mode.

        It will calculate a sample position and sample interval and set
        start index 0 when sample_pos == 1 or randomly choose from
        [0, sample_pos - 1]. Then it will shift the start index by each
        base offset.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_idx = 0 if sample_position == 1 else np.random.randint(
            0, sample_position - 1)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = (base_offsets + start_idx) % num_frames
        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in test mode.

        It will calculate a sample position and sample interval and evenly
        sample several start indexes as start positions between
        [0, sample_position-1]. Then it will shift each start index by the
        base offsets.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_list = np.linspace(
            0, sample_position - 1, num=self.num_sample_positions, dtype=int)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = list()
        for start_idx in start_list:
            clip_offsets.extend((base_offsets + start_idx) % num_frames)
        clip_offsets = np.array(clip_offsets)
        return clip_offsets

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'sample_range={self.sample_range}, '
                    f'num_sample_positions={self.num_sample_positions}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class SampleAVAFrames(SampleFrames):

    def __init__(self, clip_len, frame_interval=2, test_mode=False):

        super().__init__(clip_len, frame_interval, test_mode=test_mode)

    def _get_clips(self, center_index, skip_offsets, shot_info):
        start = center_index - (self.clip_len // 2) * self.frame_interval
        end = center_index + ((self.clip_len + 1) // 2) * self.frame_interval
        frame_inds = list(range(start, end, self.frame_interval))
        if not self.test_mode:
            frame_inds = frame_inds + skip_offsets
        frame_inds = np.clip(frame_inds, shot_info[0], shot_info[1] - 1)
        return frame_inds

    def __call__(self, results):
        fps = results['fps']
        timestamp = results['timestamp']
        timestamp_start = results['timestamp_start']
        shot_info = results['shot_info']

        center_index = fps * (timestamp - timestamp_start) + 1

        skip_offsets = np.random.randint(
            -self.frame_interval // 2, (self.frame_interval + 1) // 2,
            size=self.clip_len)
        frame_inds = self._get_clips(center_index, skip_offsets, shot_info)
        start_index = results.get('start_index', 0)

        frame_inds = np.array(frame_inds, dtype=np.int) + start_index
        results['frame_inds'] = frame_inds
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = 1
        results['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class SampleProposalFrames(SampleFrames):
    """Sample frames from proposals in the video.

    Required keys are "total_frames" and "out_proposals", added or
    modified keys are "frame_inds", "frame_interval", "num_clips",
    'clip_len' and 'num_proposals'.

    Args:
        clip_len (int): Frames of each sampled output clip.
        body_segments (int): Number of segments in course period.
        aug_segments (list[int]): Number of segments in starting and
            ending period.
        aug_ratio (int | float | tuple[int | float]): The ratio
            of the length of augmentation to that of the proposal.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        test_interval (int): Temporal interval of adjacent sampled frames
            in test mode. Default: 6.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        mode (str): Choose 'train', 'val' or 'test' mode.
            Default: 'train'.
    """

    def __init__(self,
                 clip_len,
                 body_segments,
                 aug_segments,
                 aug_ratio,
                 frame_interval=1,
                 test_interval=6,
                 temporal_jitter=False,
                 mode='train'):
        super().__init__(
            clip_len,
            frame_interval=frame_interval,
            temporal_jitter=temporal_jitter)
        self.body_segments = body_segments
        self.aug_segments = aug_segments
        self.aug_ratio = _pair(aug_ratio)
        if not mmcv.is_tuple_of(self.aug_ratio, (int, float)):
            raise TypeError(f'aug_ratio should be int, float'
                            f'or tuple of int and float, '
                            f'but got {type(aug_ratio)}')
        assert len(self.aug_ratio) == 2
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.test_interval = test_interval

    @staticmethod
    def _get_train_indices(valid_length, num_segments):
        """Get indices of different stages of proposals in train mode.

        It will calculate the average interval for each segment,
        and randomly shift them within offsets between [0, average_duration].
        If the total number of frames is smaller than num segments, it will
        return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        avg_interval = (valid_length + 1) // num_segments
        if avg_interval > 0:
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = base_offsets + np.random.randint(
                avg_interval, size=num_segments)
        else:
            offsets = np.zeros((num_segments, ), dtype=np.int)

        return offsets

    @staticmethod
    def _get_val_indices(valid_length, num_segments):
        """Get indices of different stages of proposals in validation mode.

        It will calculate the average interval for each segment.
        If the total number of valid length is smaller than num segments,
        it will return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in validation mode.
        """
        if valid_length >= num_segments:
            avg_interval = valid_length / float(num_segments)
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        else:
            offsets = np.zeros((num_segments, ), dtype=np.int)

        return offsets

    def _get_proposal_clips(self, proposal, num_frames):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices in the proposal's three
        stages: starting, course and ending stage.

        Args:
            proposal (obj): The proposal object.
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        # proposal interval: [start_frame, end_frame)
        start_frame = proposal.start_frame
        end_frame = proposal.end_frame
        ori_clip_len = self.clip_len * self.frame_interval

        duration = end_frame - start_frame
        assert duration != 0
        valid_length = duration - ori_clip_len

        valid_starting = max(0,
                             start_frame - int(duration * self.aug_ratio[0]))
        valid_ending = min(num_frames - ori_clip_len + 1,
                           end_frame - 1 + int(duration * self.aug_ratio[1]))

        valid_starting_length = start_frame - valid_starting - ori_clip_len
        valid_ending_length = (valid_ending - end_frame + 1) - ori_clip_len

        if self.mode == 'train':
            starting_offsets = self._get_train_indices(valid_starting_length,
                                                       self.aug_segments[0])
            course_offsets = self._get_train_indices(valid_length,
                                                     self.body_segments)
            ending_offsets = self._get_train_indices(valid_ending_length,
                                                     self.aug_segments[1])
        elif self.mode == 'val':
            starting_offsets = self._get_val_indices(valid_starting_length,
                                                     self.aug_segments[0])
            course_offsets = self._get_val_indices(valid_length,
                                                   self.body_segments)
            ending_offsets = self._get_val_indices(valid_ending_length,
                                                   self.aug_segments[1])
        starting_offsets += valid_starting
        course_offsets += start_frame
        ending_offsets += end_frame

        offsets = np.concatenate(
            (starting_offsets, course_offsets, ending_offsets))
        return offsets

    def _get_train_clips(self, num_frames, proposals):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices of each proposal, and then
        assemble them.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list): Proposals fetched.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        clip_offsets = []
        for proposal in proposals:
            proposal_clip_offsets = self._get_proposal_clips(
                proposal[0][1], num_frames)
            clip_offsets = np.concatenate(
                [clip_offsets, proposal_clip_offsets])

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        It will calculate sampled frame indices based on test interval.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        return np.arange(
            0, num_frames - ori_clip_len, self.test_interval, dtype=np.int)

    def _sample_clips(self, num_frames, proposals):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list | None): Proposals fetched.
                It is set to None in test mode.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.mode == 'test':
            clip_offsets = self._get_test_clips(num_frames)
        else:
            assert proposals is not None
            clip_offsets = self._get_train_clips(num_frames, proposals)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        out_proposals = results.get('out_proposals', None)
        clip_offsets = self._sample_clips(total_frames, out_proposals)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        start_index = results['start_index']
        frame_inds = np.mod(frame_inds, total_frames) + start_index

        results['frame_inds'] = np.array(frame_inds).astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = (
            self.body_segments + self.aug_segments[0] + self.aug_segments[1])
        if self.mode in ['train', 'val']:
            results['num_proposals'] = len(results['out_proposals'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'body_segments={self.body_segments}, '
                    f'aug_segments={self.aug_segments}, '
                    f'aug_ratio={self.aug_ratio}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_interval={self.test_interval}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'mode={self.mode})')
        return repr_str


@PIPELINES.register_module()
class PyAVInit:
    """Using pyav to initialize the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "filename",
    added or modified keys are "video_reader", and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', **kwargs):
        """ 已看過，使用PyAV對影片進行初始化
        Args:
            io_backend: 影片存放的設備
            kwargs: 其他參數
        """
        # 保存io_backend資料
        self.io_backend = io_backend
        # 保存kwargs當中的參數
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the PyAV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # 已看過，透過PyAV進行初始化
        try:
            # 這裡我們先嘗試將av導入
            import av
        except ImportError:
            # 如果無法導入av就會報錯，這裡會需要安裝av才可以進行
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        if self.file_client is None:
            # 獲取影片存放設備對應需要使用到的FileClient類
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        # 讀取影片，這裡會先透過file_client.get獲取影片的二進制資料，之後再透過io.BytesIO將二進制資料包裝起來
        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        # 使用ac的open將每一幀提取出來
        container = av.open(file_obj)

        # 將container保存到results當中
        results['video_reader'] = container
        # 計算當前影片總共的幀數
        results['total_frames'] = container.streams.video[0].frames

        # 回傳更新後的results
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(io_backend={self.io_backend})'
        return repr_str


@PIPELINES.register_module()
class PyAVDecode:
    """Using PyAV to decode the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            the nearest key frames, which may be duplicated and inaccurate,
            and more suitable for large scene-based video datasets.
            Default: 'accurate'.
    """

    def __init__(self, multi_thread=False, mode='accurate'):
        """ 已看過，使用PyAV對影像進行解碼
        Args:
            multi_thread: 是否使用多線程
            mode: decoding的模式
        """
        # 將傳入資料進行保存
        self.multi_thread = multi_thread
        self.mode = mode
        # mode需要是accurate或是efficient兩種
        assert mode in ['accurate', 'efficient']

    @staticmethod
    def frame_generator(container, stream):
        """Frame generator for PyAV."""
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame:
                    return frame.to_rgb().to_ndarray()

    def __call__(self, results):
        """Perform the PyAV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # 已看過，對影片進行解碼
        # 獲取container，這裡的會是透過av模組獲取的影片擷取成幀的實例對象
        container = results['video_reader']
        # 構建一個imgs的list
        imgs = list()

        if self.multi_thread:
            # 如果有啟用multi_thread就會到這裡
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            # 如果results當中的frame_inds的維度不是1就會到這裡，將frame_inds進行壓平
            # 不是1的情況就是shape為[num_clip, clip_len]我們要轉成[num_clip * clip_len]
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        if self.mode == 'accurate':
            # 如果使用的模式是accurate就會到這裡，accurate會裁切的更加精準但是花費時間較長
            # set max indices to make early stop
            # 獲取指定幀數當中最大的幀數值，表示切到這裡就可以停止
            max_inds = max(results['frame_inds'])
            # 給一個計數當前是第幾幀
            i = 0
            # 開始遍歷每一幀圖像，這裡用到的是container.decode的方式讀取
            for frame in container.decode(video=0):
                # 如果已經超過需要幀的最後一個就提前跳出
                if i > max_inds + 1:
                    break
                # 將獲取的frame轉成rgb且換成ndarray後添加到imgs當中，獲取到的就會是ndarray shape = [height, width, channel=3]
                imgs.append(frame.to_rgb().to_ndarray())
                # 將計數器多一
                i += 1

            # the available frame in pyav may be less than its length, which may raise error (應該不會發生這種問題)
            # 透過指定的幀獲取該幀的圖像
            results['imgs'] = [
                imgs[i % len(imgs)] for i in results['frame_inds']
            ]
        elif self.mode == 'efficient':
            # 如果是希望有效率的獲取圖像資料，但是精確度就不會太高，就會到這裡
            for frame in container.decode(video=0):
                backup_frame = frame
                break
            stream = container.streams.video[0]
            for idx in results['frame_inds']:
                pts_scale = stream.average_rate * stream.time_base
                frame_pts = int(idx / pts_scale)
                container.seek(
                    frame_pts, any_frame=False, backward=True, stream=stream)
                frame = self.frame_generator(container, stream)
                if frame is not None:
                    imgs.append(frame)
                    backup_frame = frame
                else:
                    imgs.append(backup_frame)
            results['imgs'] = imgs
        # 將當前圖像的大小放到results當中，這裡會是original_shape
        results['original_shape'] = imgs[0].shape[:2]
        # 將當前圖像大小也記錄下來，之後有進行reshape等操作就會改這裡的大小
        results['img_shape'] = imgs[0].shape[:2]
        # 將video_reader的av實例化對象清空
        results['video_reader'] = None
        # 將container移除，可以節省記憶體消耗
        del container

        # 回傳更新後的results
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread}, mode={self.mode})'
        return repr_str


@PIPELINES.register_module()
class PIMSInit:
    """Use PIMS to initialize the video.

    PIMS: https://github.com/soft-matter/pims

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will always use ``pims.PyAVReaderIndexed``
            to decode videos into accurate frames. If set to 'efficient', it
            will adopt fast seeking by using ``pims.PyAVReaderTimed``.
            Both will return the accurate frames in most cases.
            Default: 'accurate'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', mode='accurate', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def __call__(self, results):
        try:
            import pims
        except ImportError:
            raise ImportError('Please run "conda install pims -c conda-forge" '
                              'or "pip install pims" to install pims first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        if self.mode == 'accurate':
            container = pims.PyAVReaderIndexed(file_obj)
        else:
            container = pims.PyAVReaderTimed(file_obj)

        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(io_backend={self.io_backend}, '
                    f'mode={self.mode})')
        return repr_str


@PIPELINES.register_module()
class PIMSDecode:
    """Using PIMS to decode the videos.

    PIMS: https://github.com/soft-matter/pims

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".
    """

    def __call__(self, results):
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        imgs = [container[idx] for idx in frame_inds]

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class PyAVDecodeMotionVector(PyAVDecode):
    """Using pyav to decode the motion vectors from video.

    Reference: https://github.com/PyAV-Org/PyAV/
        blob/main/tests/test_decode.py

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "motion_vectors", "frame_inds".
    """

    @staticmethod
    def _parse_vectors(mv, vectors, height, width):
        """Parse the returned vectors."""
        (w, h, src_x, src_y, dst_x,
         dst_y) = (vectors['w'], vectors['h'], vectors['src_x'],
                   vectors['src_y'], vectors['dst_x'], vectors['dst_y'])
        val_x = dst_x - src_x
        val_y = dst_y - src_y
        start_x = dst_x - w // 2
        start_y = dst_y - h // 2
        end_x = start_x + w
        end_y = start_y + h
        for sx, ex, sy, ey, vx, vy in zip(start_x, end_x, start_y, end_y,
                                          val_x, val_y):
            if (sx >= 0 and ex < width and sy >= 0 and ey < height):
                mv[sy:ey, sx:ex] = (vx, vy)

        return mv

    def __call__(self, results):
        """Perform the PyAV motion vector decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        # set max index to make early stop
        max_idx = max(results['frame_inds'])
        i = 0
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        codec_context.options = {'flags2': '+export_mvs'}
        for packet in container.demux(stream):
            for frame in packet.decode():
                if i > max_idx + 1:
                    break
                i += 1
                height = frame.height
                width = frame.width
                mv = np.zeros((height, width, 2), dtype=np.int8)
                vectors = frame.side_data.get('MOTION_VECTORS')
                if frame.key_frame:
                    # Key frame don't have motion vectors
                    assert vectors is None
                if vectors is not None and len(vectors) > 0:
                    mv = self._parse_vectors(mv, vectors.to_ndarray(), height,
                                             width)
                imgs.append(mv)

        results['video_reader'] = None
        del container

        # the available frame in pyav may be less than its length,
        # which may raise error
        results['motion_vectors'] = np.array(
            [imgs[i % len(imgs)] for i in results['frame_inds']])
        return results


@PIPELINES.register_module()
class DecordInit:
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        num_threads (int): Number of thread to decode the video. Default: 1.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', num_threads=1, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the Decord initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        results['video_reader'] = container
        results['total_frames'] = len(container)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@PIPELINES.register_module()
class DecordDecode:
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".

    Args:
        mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
            If set to 'accurate', it will decode videos into accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return
            key frames, which may be duplicated and inaccurate, and more
            suitable for large scene-based video datasets. Default: 'accurate'.
    """

    def __init__(self, mode='accurate'):
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    def __call__(self, results):
        """Perform the Decord decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']

        if self.mode == 'accurate':
            imgs = container.get_batch(frame_inds).asnumpy()
            imgs = list(imgs)
        elif self.mode == 'efficient':
            # This mode is faster, however it always returns I-FRAME
            container.seek(0)
            imgs = list()
            for idx in frame_inds:
                container.seek(idx)
                frame = container.next()
                imgs.append(frame.asnumpy())

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(mode={self.mode})'
        return repr_str


@PIPELINES.register_module()
class OpenCVInit:
    """Using OpenCV to initialize the video_reader.

    Required keys are "filename", added or modified keys are "new_path",
    "video_reader" and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        self.tmp_folder = None
        if self.io_backend != 'disk':
            random_string = get_random_string()
            thread_id = get_thread_id()
            self.tmp_folder = osp.join(get_shm_dir(),
                                       f'{random_string}_{thread_id}')
            os.mkdir(self.tmp_folder)

    def __call__(self, results):
        """Perform the OpenCV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.io_backend == 'disk':
            new_path = results['filename']
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend, **self.kwargs)

            thread_id = get_thread_id()
            # save the file of same thread at the same place
            new_path = osp.join(self.tmp_folder, f'tmp_{thread_id}.mp4')
            with open(new_path, 'wb') as f:
                f.write(self.file_client.get(results['filename']))

        container = mmcv.VideoReader(new_path)
        results['new_path'] = new_path
        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __del__(self):
        if self.tmp_folder and osp.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend})')
        return repr_str


@PIPELINES.register_module()
class OpenCVDecode:
    """Using OpenCV to decode the video.

    Required keys are "video_reader", "filename" and "frame_inds", added or
    modified keys are "imgs", "img_shape" and "original_shape".
    """

    def __call__(self, results):
        """Perform the OpenCV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_ind in results['frame_inds']:
            cur_frame = container[frame_ind]
            # last frame may be None in OpenCV
            while isinstance(cur_frame, type(None)):
                frame_ind -= 1
                cur_frame = container[frame_ind]
            imgs.append(cur_frame)

        results['video_reader'] = None
        del container

        imgs = np.array(imgs)
        # The default channel order of OpenCV is BGR, thus we change it to RGB
        imgs = imgs[:, :, :, ::-1]
        results['imgs'] = list(imgs)
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class RawFrameDecode:
    """Load and decode frames with given indices.

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        cache = {}
        for i, frame_idx in enumerate(results['frame_inds']):
            # Avoid loading duplicated frames
            if frame_idx in cache:
                if modality == 'RGB':
                    imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                else:
                    imgs.append(cp.deepcopy(imgs[2 * cache[frame_idx]]))
                    imgs.append(cp.deepcopy(imgs[2 * cache[frame_idx] + 1]))
                continue
            else:
                cache[frame_idx] = i

            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str


@PIPELINES.register_module()
class ArrayDecode:
    """Load and decode frames with given indices from a 4D array.

    Required keys are "array and "frame_inds", added or modified keys are
    "imgs", "img_shape" and "original_shape".
    """

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        modality = results['modality']
        array = results['array']

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for i, frame_idx in enumerate(results['frame_inds']):

            frame_idx += offset
            if modality == 'RGB':
                imgs.append(array[frame_idx])
            elif modality == 'Flow':
                imgs.extend(
                    [array[frame_idx, ..., 0], array[frame_idx, ..., 1]])
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@PIPELINES.register_module()
class ImageDecode:
    """Load and decode images.

    Required key is "filename", added or modified keys are "imgs", "img_shape"
    and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``ImageDecode`` to load image given the file path.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        filename = results['filename']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()
        img_bytes = self.file_client.get(filename)

        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        imgs.append(img)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        return results


@PIPELINES.register_module()
class AudioDecodeInit:
    """Using librosa to initialize the audio reader.

    Required keys are "audio_path", added or modified keys are "length",
    "sample_rate", "audios".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        sample_rate (int): Audio sampling times per second. Default: 16000.
    """

    def __init__(self,
                 io_backend='disk',
                 sample_rate=16000,
                 pad_method='zero',
                 **kwargs):
        self.io_backend = io_backend
        self.sample_rate = sample_rate
        if pad_method in ['random', 'zero']:
            self.pad_method = pad_method
        else:
            raise NotImplementedError
        self.kwargs = kwargs
        self.file_client = None

    @staticmethod
    def _zero_pad(shape):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def _random_pad(shape):
        # librosa load raw audio file into a distribution of -1~+1
        return np.random.rand(shape).astype(np.float32) * 2 - 1

    def __call__(self, results):
        """Perform the librosa initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import librosa
        except ImportError:
            raise ImportError('Please install librosa first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        if osp.exists(results['audio_path']):
            file_obj = io.BytesIO(self.file_client.get(results['audio_path']))
            y, sr = librosa.load(file_obj, sr=self.sample_rate)
        else:
            # Generate a random dummy 10s input
            pad_func = getattr(self, f'_{self.pad_method}_pad')
            y = pad_func(int(round(10.0 * self.sample_rate)))
            sr = self.sample_rate

        results['length'] = y.shape[0]
        results['sample_rate'] = sr
        results['audios'] = y
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'sample_rate={self.sample_rate}, '
                    f'pad_method={self.pad_method})')
        return repr_str


@PIPELINES.register_module()
class LoadAudioFeature:
    """Load offline extracted audio features.

    Required keys are "audio_path", added or modified keys are "length",
    audios".
    """

    def __init__(self, pad_method='zero'):
        if pad_method not in ['zero', 'random']:
            raise NotImplementedError
        self.pad_method = pad_method

    @staticmethod
    def _zero_pad(shape):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def _random_pad(shape):
        # spectrogram is normalized into a distribution of 0~1
        return np.random.rand(shape).astype(np.float32)

    def __call__(self, results):
        """Perform the numpy loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if osp.exists(results['audio_path']):
            feature_map = np.load(results['audio_path'])
        else:
            # Generate a random dummy 10s input
            # Some videos do not have audio stream
            pad_func = getattr(self, f'_{self.pad_method}_pad')
            feature_map = pad_func((640, 80))

        results['length'] = feature_map.shape[0]
        results['audios'] = feature_map
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'pad_method={self.pad_method})')
        return repr_str


@PIPELINES.register_module()
class AudioDecode:
    """Sample the audio w.r.t. the frames selected.

    Args:
        fixed_length (int): As the audio clip selected by frames sampled may
            not be exactly the same, `fixed_length` will truncate or pad them
            into the same size. Default: 32000.

    Required keys are "frame_inds", "num_clips", "total_frames", "length",
    added or modified keys are "audios", "audios_shape".
    """

    def __init__(self, fixed_length=32000):
        self.fixed_length = fixed_length

    def __call__(self, results):
        """Perform the ``AudioDecode`` to pick audio clips."""
        audio = results['audios']
        frame_inds = results['frame_inds']
        num_clips = results['num_clips']
        resampled_clips = list()
        frame_inds = frame_inds.reshape(num_clips, -1)
        for clip_idx in range(num_clips):
            clip_frame_inds = frame_inds[clip_idx]
            start_idx = max(
                0,
                int(
                    round((clip_frame_inds[0] + 1) / results['total_frames'] *
                          results['length'])))
            end_idx = min(
                results['length'],
                int(
                    round((clip_frame_inds[-1] + 1) / results['total_frames'] *
                          results['length'])))
            cropped_audio = audio[start_idx:end_idx]
            if cropped_audio.shape[0] >= self.fixed_length:
                truncated_audio = cropped_audio[:self.fixed_length]
            else:
                truncated_audio = np.pad(
                    cropped_audio,
                    ((0, self.fixed_length - cropped_audio.shape[0])),
                    mode='constant')

            resampled_clips.append(truncated_audio)

        results['audios'] = np.array(resampled_clips)
        results['audios_shape'] = results['audios'].shape
        return results


@PIPELINES.register_module()
class BuildPseudoClip:
    """Build pseudo clips with one single image by repeating it n times.

    Required key is "imgs", added or modified key is "imgs", "num_clips",
        "clip_len".

    Args:
        clip_len (int): Frames of the generated pseudo clips.
    """

    def __init__(self, clip_len):
        self.clip_len = clip_len

    def __call__(self, results):
        # the input should be one single image
        assert len(results['imgs']) == 1
        im = results['imgs'][0]
        for _ in range(1, self.clip_len):
            results['imgs'].append(np.copy(im))
        results['clip_len'] = self.clip_len
        results['num_clips'] = 1
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'fix_length={self.fixed_length})')
        return repr_str


@PIPELINES.register_module()
class AudioFeatureSelector:
    """Sample the audio feature w.r.t. the frames selected.

    Required keys are "audios", "frame_inds", "num_clips", "length",
    "total_frames", added or modified keys are "audios", "audios_shape".

    Args:
        fixed_length (int): As the features selected by frames sampled may
            not be exactly the same, `fixed_length` will truncate or pad them
            into the same size. Default: 128.
    """

    def __init__(self, fixed_length=128):
        self.fixed_length = fixed_length

    def __call__(self, results):
        """Perform the ``AudioFeatureSelector`` to pick audio feature clips.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        audio = results['audios']
        frame_inds = results['frame_inds']
        num_clips = results['num_clips']
        resampled_clips = list()

        frame_inds = frame_inds.reshape(num_clips, -1)
        for clip_idx in range(num_clips):
            clip_frame_inds = frame_inds[clip_idx]
            start_idx = max(
                0,
                int(
                    round((clip_frame_inds[0] + 1) / results['total_frames'] *
                          results['length'])))
            end_idx = min(
                results['length'],
                int(
                    round((clip_frame_inds[-1] + 1) / results['total_frames'] *
                          results['length'])))
            cropped_audio = audio[start_idx:end_idx, :]
            if cropped_audio.shape[0] >= self.fixed_length:
                truncated_audio = cropped_audio[:self.fixed_length, :]
            else:
                truncated_audio = np.pad(
                    cropped_audio,
                    ((0, self.fixed_length - cropped_audio.shape[0]), (0, 0)),
                    mode='constant')

            resampled_clips.append(truncated_audio)
        results['audios'] = np.array(resampled_clips)
        results['audios_shape'] = results['audios'].shape
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'fix_length={self.fixed_length})')
        return repr_str


@PIPELINES.register_module()
class LoadLocalizationFeature:
    """Load Video features for localizer with given video_name list.

    Required keys are "video_name" and "data_prefix", added or modified keys
    are "raw_feature".

    Args:
        raw_feature_ext (str): Raw feature file extension.  Default: '.csv'.
    """

    def __init__(self, raw_feature_ext='.csv'):
        valid_raw_feature_ext = ('.csv', )
        if raw_feature_ext not in valid_raw_feature_ext:
            raise NotImplementedError
        self.raw_feature_ext = raw_feature_ext

    def __call__(self, results):
        """Perform the LoadLocalizationFeature loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_name = results['video_name']
        data_prefix = results['data_prefix']

        data_path = osp.join(data_prefix, video_name + self.raw_feature_ext)
        raw_feature = np.loadtxt(
            data_path, dtype=np.float32, delimiter=',', skiprows=1)

        results['raw_feature'] = np.transpose(raw_feature, (1, 0))

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'raw_feature_ext={self.raw_feature_ext})')
        return repr_str


@PIPELINES.register_module()
class GenerateLocalizationLabels:
    """Load video label for localizer with given video_name list.

    Required keys are "duration_frame", "duration_second", "feature_frame",
    "annotations", added or modified keys are "gt_bbox".
    """

    def __call__(self, results):
        """Perform the GenerateLocalizationLabels loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_frame = results['duration_frame']
        video_second = results['duration_second']
        feature_frame = results['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second
        annotations = results['annotations']

        gt_bbox = []

        for annotation in annotations:
            current_start = max(
                min(1, annotation['segment'][0] / corrected_second), 0)
            current_end = max(
                min(1, annotation['segment'][1] / corrected_second), 0)
            gt_bbox.append([current_start, current_end])

        gt_bbox = np.array(gt_bbox)
        results['gt_bbox'] = gt_bbox
        return results


@PIPELINES.register_module()
class LoadProposals:
    """Loading proposals with given proposal results.

    Required keys are "video_name", added or modified keys are 'bsp_feature',
    'tmin', 'tmax', 'tmin_score', 'tmax_score' and 'reference_temporal_iou'.

    Args:
        top_k (int): The top k proposals to be loaded.
        pgm_proposals_dir (str): Directory to load proposals.
        pgm_features_dir (str): Directory to load proposal features.
        proposal_ext (str): Proposal file extension. Default: '.csv'.
        feature_ext (str): Feature file extension. Default: '.npy'.
    """

    def __init__(self,
                 top_k,
                 pgm_proposals_dir,
                 pgm_features_dir,
                 proposal_ext='.csv',
                 feature_ext='.npy'):
        self.top_k = top_k
        self.pgm_proposals_dir = pgm_proposals_dir
        self.pgm_features_dir = pgm_features_dir
        valid_proposal_ext = ('.csv', )
        if proposal_ext not in valid_proposal_ext:
            raise NotImplementedError
        self.proposal_ext = proposal_ext
        valid_feature_ext = ('.npy', )
        if feature_ext not in valid_feature_ext:
            raise NotImplementedError
        self.feature_ext = feature_ext

    def __call__(self, results):
        """Perform the LoadProposals loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_name = results['video_name']
        proposal_path = osp.join(self.pgm_proposals_dir,
                                 video_name + self.proposal_ext)
        if self.proposal_ext == '.csv':
            pgm_proposals = np.loadtxt(
                proposal_path, dtype=np.float32, delimiter=',', skiprows=1)

        pgm_proposals = np.array(pgm_proposals[:self.top_k])
        tmin = pgm_proposals[:, 0]
        tmax = pgm_proposals[:, 1]
        tmin_score = pgm_proposals[:, 2]
        tmax_score = pgm_proposals[:, 3]
        reference_temporal_iou = pgm_proposals[:, 5]

        feature_path = osp.join(self.pgm_features_dir,
                                video_name + self.feature_ext)
        if self.feature_ext == '.npy':
            bsp_feature = np.load(feature_path).astype(np.float32)

        bsp_feature = bsp_feature[:self.top_k, :]

        results['bsp_feature'] = bsp_feature
        results['tmin'] = tmin
        results['tmax'] = tmax
        results['tmin_score'] = tmin_score
        results['tmax_score'] = tmax_score
        results['reference_temporal_iou'] = reference_temporal_iou

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'top_k={self.top_k}, '
                    f'pgm_proposals_dir={self.pgm_proposals_dir}, '
                    f'pgm_features_dir={self.pgm_features_dir}, '
                    f'proposal_ext={self.proposal_ext}, '
                    f'feature_ext={self.feature_ext})')
        return repr_str
