import copy
import os
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.ST.utils import get_cls_from_dict


class ObjectDetection:
    def __init__(self, cfg_path):
        from SpecialTopic.WorkingFlow.SubBlock.ObjectDetection.YoloxObjectDetection import YoloxObjectDetection
        support_module = {
            'YoloxObjectDetection': YoloxObjectDetection
        }
        cfg = parser_cfg(cfg_path)
        cfg_ = copy.deepcopy(cfg)
        self.cfg = cfg
        module_cls = get_cls_from_dict(support_module, cfg_)
        if 'pretrained' in cfg_.keys():
            pretrained = cfg_.get('pretrained')
            if not os.path.isfile(pretrained):
                cfg_['pretrained'] = 'none'
        self.module = module_cls(**cfg_)

    def __call__(self, call_api, inputs):
        results = self.module(call_api, inputs)
        return results
