import numpy as np
from ..utils import get_cls_from_dict


class Compose:
    def __init__(self, pipelines_cfg):
        from .pipeline import LoadInfoFromAnno, Resize, ResizeAndAugmentation, Mosaic, Collect
        support_pipeline = {
            'LoadInfoFromAnno': LoadInfoFromAnno,
            'Resize': Resize,
            'ResizeAndAugmentation': ResizeAndAugmentation,
            'Mosaic': Mosaic,
            'Collect': Collect
        }
        self.pipelines = list()
        for pipeline_cfg in pipelines_cfg:
            pipeline_cls = get_cls_from_dict(support_pipeline, pipeline_cfg)
            pipeline = pipeline_cls(**pipeline_cfg)
            self.pipelines.append(pipeline)

    def __call__(self, data):
        for pipeline in self.pipelines:
            data = pipeline(data)
        return data


def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a
