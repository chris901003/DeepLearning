import copy
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.ST.utils import get_cls_from_dict


class ShowResults:
    def __init__(self, cfg_file):
        from SpecialTopic.WorkingFlow.SubBlock.ShowResults.DrawResultsOnPicture import DrawResultsOnPicture
        support_module = {
            'DrawResultsOnPicture': DrawResultsOnPicture
        }
        cfg = parser_cfg(cfg_file)
        cfg_ = copy.deepcopy(cfg)
        self.cfg = cfg
        module_cls = get_cls_from_dict(support_module, cfg_)
        self.module = module_cls(**cfg_)

    def __call__(self, call_api, inputs):
        results = self.module(call_api, inputs)
        return results
