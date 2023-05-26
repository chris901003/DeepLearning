import copy
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.ST.utils import get_cls_from_dict


class RemainDetection:
    def __init__(self, cfg_path, logger):
        """ 進行剩餘量檢測的主模塊部分
        Args:
             cfg_path: 子模塊配置文件
             logger: 保存運行過程中狀態的log實例化對象
        """
        from SpecialTopic.WorkingFlow.SubBlock.RemainDetection.VitRemainDetection import VitRemainDetection
        from SpecialTopic.WorkingFlow.SubBlock.RemainDetection.SegformerRemainDetection import SegformerRemainDetection
        from SpecialTopic.WorkingFlow.SubBlock.RemainDetection.SegformerRemainDetectionTensorRT import \
            SegformerRemainDetectionTensorRT
        from SpecialTopic.WorkingFlow.SubBlock.RemainDetection.SegformerWithDeepRemainDetection import \
            SegformerWithDeepRemainDetection
        from SpecialTopic.WorkingFlow.SubBlock.RemainDetection.SegformerWithDeepRemainDetectionTensorRT import \
            SegformerWithDeepRemainDetectionTensorRT
        from SpecialTopic.WorkingFlow.SubBlock.RemainDetection.SegformerWithDeepV2 import SegformerWithDeepV2
        support_module = {
            'VitRemainDetection': VitRemainDetection,
            'SegformerRemainDetection': SegformerRemainDetection,
            'SegformerRemainDetectionTensorRT': SegformerRemainDetectionTensorRT,
            'SegformerWithDeepRemainDetection': SegformerWithDeepRemainDetection,
            'SegformerWithDeepRemainDetectionTensorRT': SegformerWithDeepRemainDetectionTensorRT,
            'SegformerWithDeepV2': SegformerWithDeepV2
        }
        cfg = parser_cfg(cfg_path)
        cfg_ = copy.deepcopy(cfg)
        self.cfg = cfg
        module_cls = get_cls_from_dict(support_module, cfg_)
        self.module = module_cls(**cfg_)
        # 將logger實例化會對象放到module當中
        self.module.logger = logger
        module_name = cfg.get('type', None)
        logger['logger'].debug(f'RemainDetection using [ {module_name} ] module')

    def __call__(self, call_api, inputs):
        results = self.module(call_api, inputs)
        return results
