import copy
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.ST.utils import get_cls_from_dict


class RemainTimeDetection:
    def __init__(self, cfg_path):
        from SpecialTopic.WorkingFlow.SubBlock.RemainTime.RemainTimeTransformerDetection \
            import RemainTimeTransformerDetection
        support_module = {
            'RemainTimeTransformerDetection': RemainTimeTransformerDetection
        }
        cfg = parser_cfg(cfg_path)
        cfg_ = copy.deepcopy(cfg)
        self.cfg = cfg
        module_cls = get_cls_from_dict(support_module, cfg_)
        self.module = module_cls(**cfg_)

    def __call__(self, call_api, inputs):
        results = self.module(call_api, inputs)
        return results
