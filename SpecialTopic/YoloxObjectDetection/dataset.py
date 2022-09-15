import copy
import os
from torch.utils.data.dataset import Dataset
from utils import get_specified_option
from pipeline import LoadImageFromFile, RandomAffine, MixUp, YOLOXHSVRandomAug, RandomFlip, Resize, Pad, \
    FilterAnnotations, Collect, LoadAnnotations, Mosaic


def build_dataset(dataset_cfg):
    support_dataset = {
        'LabelImgYoloFormat': LabelImgYoloFormat,
        'MultiImageMixDataset': MultiImageMixDataset
    }
    dataset_cls = get_specified_option(support_dataset, dataset_cfg)
    dataset = dataset_cls(**dataset_cfg)
    return dataset


class MultiImageMixDataset(Dataset):
    def __init__(self, dataset, pipelines, max_fetch=15):
        support_pipeline = {
            'Mosaic': Mosaic,
            'RandomAffine': RandomAffine,
            'MixUp': MixUp,
            'YOLOXHSVRandomAug': YOLOXHSVRandomAug,
            'RandomFlip': RandomFlip,
            'Resize': Resize,
            'Pad': Pad,
            'FilterAnnotations': FilterAnnotations,
            'Collect': Collect
        }
        self.dataset = build_dataset(dataset)
        self.num_samples = len(self.dataset)
        self.pipelines = list()
        self.max_fetch = max_fetch
        for pipeline_cfg in pipelines:
            if isinstance(pipeline_cfg, dict):
                pipeline_cls = get_specified_option(support_pipeline, pipeline_cfg)
                pipeline = pipeline_cls(**pipeline_cfg)
                self.pipelines.append(pipeline)

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for pipeline in self.pipelines:
            if hasattr(pipeline, 'get_indexes'):
                for _ in range(self.max_fetch):
                    indexes = pipeline.get_indexes(self.dataset)
                    if not isinstance(indexes, list):
                        indexes = [indexes]
                    mix_results = [copy.deepcopy(self.dataset[index]) for index in indexes]
                    if None not in mix_results:
                        results['mix_results'] = mix_results
                        break
                else:
                    raise RuntimeError('採樣過多次沒有任何目標')
            for _ in range(self.max_fetch):
                updated_results = pipeline(copy.deepcopy(results))
                if updated_results is not None:
                    break
            else:
                raise RuntimeError('採樣過多次')

            if 'mix_results' in results:
                results.pop('mix_results')
        return results

    def __len__(self):
        return self.num_samples


class LabelImgYoloFormat(Dataset):
    def __init__(self, annotations, images, pipelines):
        self.annotations = annotations
        self.images = images
        self.pipelines = pipelines
        self.data_info = self.prepare_data_info()
        self.pipelines = Compose(pipelines)

    def prepare_data_info(self):
        support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
        results = list()
        assert os.path.isdir(self.images)
        for image_name in os.listdir(self.images):
            if os.path.splitext(image_name)[1] not in support_image_format:
                continue
            image_path = os.path.join(self.images, image_name)
            annotation_name = os.path.splitext(image_name)[0] + '.txt'
            annotation_path = os.path.join(self.annotations, annotation_name)
            print(annotation_path)
            assert os.path.isfile(annotation_path)
            data = {
                'image_path': image_path,
                'annotation_path': annotation_path
            }
            results.append(data)
        return results

    def __getitem__(self, idx):
        data = self.data_info[idx]
        data = self.pipelines(data)
        return data

    def __len__(self):
        return len(self.data_info)


class Compose:
    def __init__(self, pipelines_cfg):
        support_pipeline = {
            'LoadImageFromFile': LoadImageFromFile,
            'LoadAnnotations': LoadAnnotations,
            'Mosaic': Mosaic,
            'RandomAffine': RandomAffine,
            'MixUp': MixUp,
            'YOLOXHSVRandomAug': YOLOXHSVRandomAug,
            'RandomFlip': RandomFlip,
            'Resize': Resize,
            'Pad': Pad,
            'FilterAnnotations': FilterAnnotations,
            'Collect': Collect
        }
        self.pipelines = list()
        for pipeline_cfg in pipelines_cfg:
            pipeline_cls = get_specified_option(support_pipeline, pipeline_cfg)
            pipeline = pipeline_cls(**pipeline_cfg)
            self.pipelines.append(pipeline)

    def __call__(self, data):
        for pipeline in self.pipelines:
            data = pipeline(data)
        return data
