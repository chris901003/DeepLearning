# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Loading image(s) from file.

    Required key: "image_file".

    Added key: "img".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        """ 從資料夾讀取圖像
        Args:
            to_float32: 是否轉成float32格式
            color_type: 圖像的顏色模式
            channel_order: channel排列順序
            file_client_args: 圖像存放設備
        """
        # 將傳入資料進行保存
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _read_image(self, path):
        # 根據指定的圖像路徑進行圖像讀取
        # 先讀取成2進制檔案
        img_bytes = self.file_client.get(path)
        # 將2進制檔案讀取成ndarray
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if img is None:
            # 如果img是None就會報錯
            raise ValueError(f'Fail to read {path}')
        if self.to_float32:
            # 如果需要轉成float32就會在這裡轉換
            img = img.astype(np.float32)
        # 回傳讀取好的圖像
        return img

    @staticmethod
    def _bgr2rgb(img):
        # 將bgr轉成rgb
        if img.ndim == 3:
            return mmcv.bgr2rgb(img)
        elif img.ndim == 4:
            return np.concatenate([mmcv.bgr2rgb(img_) for img_ in img], axis=0)
        else:
            raise ValueError('results["img"] has invalid shape '
                             f'{img.shape}')

    def __call__(self, results):
        """Loading image(s) from file."""
        # 將圖像從資料夾當中讀取出來
        if self.file_client is None:
            # 如果沒有設定讀取圖像的方式就會到這裡
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # 獲取要讀取圖像的檔案位置
        image_file = results.get('image_file', None)

        if isinstance(image_file, (list, tuple)):
            # 如果image_file是list或是tuple型態就會到這裡，使用遍歷的方式進行圖像讀取
            # Load images from a list of paths
            results['img'] = [self._read_image(path) for path in image_file]
        elif image_file is not None:
            # Load single image from path
            # 進行單張讀取
            results['img'] = self._read_image(image_file)
        else:
            # 其他情況就會到這裡
            if 'img' not in results:
                # 如果results當中沒有img資訊就會報錯
                # If `image_file`` is not in results, check the `img` exists
                # and format the image. This for compatibility when the image
                # is manually set outside the pipeline.
                raise KeyError('Either `image_file` or `img` should exist in '
                               'results.')
            if isinstance(results['img'], (list, tuple)):
                # 如果results當中有img且是list或是tuple就需要已經是ndarray格式
                assert isinstance(results['img'][0], np.ndarray)
            else:
                # 其他就直接判斷results當中img是否為ndarray
                assert isinstance(results['img'], np.ndarray)
            if self.color_type == 'color' and self.channel_order == 'rgb':
                # 如果圖像是color且channel順序是rgb就會到這裡
                # The original results['img'] is assumed to be image(s) in BGR
                # order, so we convert the color according to the arguments.
                # 將圖像從bgr轉成rgb通道
                if isinstance(results['img'], (list, tuple)):
                    results['img'] = [
                        self._bgr2rgb(img) for img in results['img']
                    ]
                else:
                    results['img'] = self._bgr2rgb(results['img'])
            # 將image_file設定成None
            results['image_file'] = None

        # 回傳更新後的results
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadVideoFromFile:
    """Loading video(s) from file.

    Required key: "video_file".

    Added key: "video".

    Args:
        to_float32 (bool): Whether to convert the loaded video to a float32
            numpy array. If set to False, the loaded video is an uint8 array.
            Defaults to False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _read_video(self, path):
        container = mmcv.VideoReader(path)
        sample = dict(
            height=int(container.height),
            width=int(container.width),
            fps=int(container.fps),
            num_frames=int(container.frame_cnt),
            video=[])
        for _ in range(container.frame_cnt):
            sample['video'].append(container.read())
        sample['video'] = np.stack(sample['video'], axis=0)
        return sample

    def __call__(self, results):
        """Loading video(s) from file."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        video_file = results.get('video_file', None)

        if isinstance(video_file, (list, tuple)):
            # Load videos from a list of paths
            for path in video_file:
                video = self._read_video(path)
                for key in video:
                    results[key].append(video[key])
        elif video_file is not None:
            # Load single video from path
            results.update(self._read_video(video_file))
        else:
            if 'video' not in results:
                # If `video_file`` is not in results, check the `video` exists
                # and format the image. This for compatibility when the image
                # is manually set outside the pipeline.
                raise KeyError('Either `video_file` or `video` should exist '
                               'in results.')
            if isinstance(results['video'], (list, tuple)):
                assert isinstance(results['video'][0], np.ndarray)
            else:
                assert isinstance(results['video'], np.ndarray)
                results['video'] = [results['video']]

            results['num_frames'] = [v.shape[0] for v in results['video']]
            results['height'] = [v.shape[1] for v in results['video']]
            results['width'] = [v.shape[2] for v in results['video']]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'file_client_args={self.file_client_args})')
        return repr_str
