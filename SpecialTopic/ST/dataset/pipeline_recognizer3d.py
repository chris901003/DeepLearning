import numpy as np
import random
from .utils import rescale_size, imresize, imflip_, imnormalize_, to_tensor


class PyAVInit:
    def __init__(self):
        pass

    def __call__(self, results):
        try:
            import av
        except ImportError:
            raise ImportError('pip install av 即可解決問題')
        video_path = results.get('filename', None)
        assert video_path is not None, '缺少影片路徑'
        container = av.open(video_path)
        results['video_reader'] = container
        results['total_frames'] = container.streams.video[0].frames
        return results


class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1, out_of_bound_opt='loop', test_mode=False):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offset = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offset + avg_interval / 2.0).astype(np.int)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)
        return clip_offsets

    def __call__(self, results):
        total_frames = results['total_frames']
        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(self.clip_len)[None, :] * self.frame_interval
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('不合法操做對於out_of_bound')
        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results


class PyAVDecode:
    def __init__(self, mode='accurate'):
        self.mode = mode
        assert mode in ['accurate', 'efficient']

    @staticmethod
    def frame_generator(container, stream):
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame:
                    return frame.to_rgb().to_ndarray()

    def __call__(self, results):
        container = results['video_reader']
        imgs = list()
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])
        if self.mode == 'accurate':
            max_inds = max(results['frame_inds'])
            i = 0
            for frame in container.decode(video=0):
                if i > max_inds + 1:
                    break
                imgs.append(frame.to_rgb().to_ndarray())
                i += 1
            results['imgs'] = [imgs[i % len(imgs)] for i in results['frame_inds']]
        elif self.mode == 'efficient':
            for frame in container.decode(video=0):
                backup_frame = frame
                break
            stream = container.streams.video[0]
            for idx in results['frame_inds']:
                pts_scale = stream.average_rate * stream.time_base
                frame_pts = int(idx / pts_scale)
                container.seek(frame_pts, any_frame=False, backward=True, stream=stream)
                frame = self.frame_generator(container, stream)
                if frame is not None:
                    imgs.append(frame)
                    backup_frame = frame
                else:
                    imgs.append(backup_frame)
            results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        results['video_reader'] = None
        del container
        return results


class Resize:
    def __init__(self, scale, keep_ratio=True, interpolation='bilinear'):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError('Scale不可以小於0')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def _resize_imgs(self, imgs, new_w, new_h):
        return [imresize(img, (new_w, new_h), interpolation=self.interpolation) for img in imgs]

    def __call__(self, results):
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']
        if self.keep_ratio:
            new_w, new_h = rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale
        self.scale_factor = np.array([new_w / img_w, new_h / img_h], dtype=np.float32)
        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor
        if 'imgs' in results:
            results['imgs'] = self._resize_imgs(results['imgs'], new_w, new_h)
        return results


class MultiScaleCrop:
    def __init__(self, input_size, scales=(1, ), max_wh_scale_gap=1, random_crop=False, num_fixed_crops=5):
        self.input_size = (input_size, input_size)
        assert num_fixed_crops in [5, 13]
        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop
        self.num_fix_crops = num_fixed_crops

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    def __call__(self, results):
        img_h, img_w = results['img_shape']
        base_size = min(img_h, img_w)
        crop_sizes = [int(base_size * s) for s in self.scales]
        candidate_sizes = list()
        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    candidate_sizes.append([w, h])
        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]
        crop_w, crop_h = crop_size
        if self.random_crop:
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            candidate_offsets = [
                (0, 0), (4 * w_step, 0), (0, 4 * h_step), (4 * w_step, 4 * h_step), (2 * w_step, 2 * h_step)
            ]
            if self.num_fix_crops == 13:
                extra_candidate_offsets = [
                    (0, 2 * h_step),
                    (4 * w_step, 2 * h_step),
                    (2 * w_step, 4 * h_step),
                    (2 * w_step, 0 * h_step),
                    (1 * w_step, 1 * h_step),
                    (3 * w_step, 1 * h_step),
                    (1 * w_step, 3 * h_step),
                    (3 * w_step, 3 * h_step)
                ]
                candidate_offsets.extend(extra_candidate_offsets)
            x_offset, y_offset = random.choice(candidate_offsets)
        new_h, new_w = crop_h, crop_w
        crop_bbox = np.array([x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)
        results['scales'] = self.scales
        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32)
        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h
        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio, old_y_ratio + y_ratio * old_h_ratio,
            w_ratio * old_w_ratio, h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(new_crop_quadruple, dtype=np.float32)
        results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        return results


class Flip:
    def __init__(self, flip_ratio=0.5, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction

    def _flip_imgs(self, imgs, modality):
        _ = [imflip_(img, self.direction) for img in imgs]
        if modality == 'Flow':
            raise NotImplementedError('目前不支持光流')
        return imgs

    def __call__(self, results):
        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'
        flip = np.random.rand() < self.flip_ratio
        results['flip'] = flip
        results['flip_direction'] = self.direction
        if flip:
            if 'imgs' in results:
                results['imgs'] = self._flip_imgs(results['imgs'], modality)
        return results


class Normalize:
    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results['modality']
        if modality == 'RGB':
            n = len(results['imgs'])
            h, w, c = results['imgs'][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs']):
                imgs[i] = img
            for img in imgs:
                imnormalize_(img, self.mean, self.std, self.to_bgr)
            results['imgs'] = imgs
            results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        if modality == 'Flow':
            raise NotImplementedError('目前不支持光流')


class FormatShape:
    def __init__(self, input_format, collapse=False):
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError('Not support format shape')

    def __call__(self, results):
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        imgs = results['imgs']
        if self.collapse:
            assert results['num_clips'] == 1
        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        elif self.input_format == 'NCHW':
            imgs = np.transpose(imgs, (0, 3, 1, 2))
        elif self.input_format == 'NCHW_Flow':
            raise NotImplementedError('目前不支持光流')
        elif self.input_format == 'NPTCHW':
            raise NotImplementedError
        if self.collapse:
            assert imgs.shape[0] == 1
            imgs = imgs.squeeze(0)
        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        return results


class ToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results
