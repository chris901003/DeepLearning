import copy
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.ST.utils import get_cls_from_dict


class ShowResults:
    def __init__(self, cfg_file, logger):
        """ 將結果顯示出來
        Args:
            cfg_file: 子模塊配置文件
            logger: 保存運行過程中狀態的log實例化對象
        """
        from SpecialTopic.WorkingFlow.SubBlock.ShowResults.DrawResultsOnPicture import DrawResultsOnPicture
        support_module = {
            'DrawResultsOnPicture': DrawResultsOnPicture
        }
        cfg = parser_cfg(cfg_file)
        cfg_ = copy.deepcopy(cfg)
        self.cfg = cfg
        module_cls = get_cls_from_dict(support_module, cfg_)
        self.module = module_cls(**cfg_)
        # 將logger實例化對象放到module上
        self.module.logger = logger
        module_name = cfg.get('type', None)
        logger['logger'].debug(f'ShowResults using [ {module_name} ] module')

    def __call__(self, call_api, inputs):
        results = self.module(call_api, inputs)
        return results
