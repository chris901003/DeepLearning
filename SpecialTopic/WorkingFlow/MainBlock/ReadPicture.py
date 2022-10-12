import copy
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.ST.utils import get_cls_from_dict


class ReadPicture:
    def __init__(self, cfg_path):
        from SpecialTopic.WorkingFlow.SubBlock.ReadPicture.ReadPictureFromVideo import ReadPictureFromVideo
        from SpecialTopic.WorkingFlow.SubBlock.ReadPicture.ReadPictureFromCamera import ReadPictureFromCamera
        from SpecialTopic.WorkingFlow.SubBlock.ReadPicture.ReadPictureFromPi import ReadPictureFromPi
        support_module = {
            'ReadPictureFromVideo': ReadPictureFromVideo,
            'ReadPictureFromCamera': ReadPictureFromCamera,
            'ReadPictureFromPi': ReadPictureFromPi
        }
        cfg = parser_cfg(cfg_path)
        self.cfg = cfg
        cfg_ = copy.deepcopy(cfg)
        module_cls = get_cls_from_dict(support_module, cfg_)
        self.module = module_cls(**cfg_)

    def __call__(self, call_api, inputs):
        results = self.module(call_api, inputs)
        return results
