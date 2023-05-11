import copy
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.ST.utils import get_cls_from_dict


class RemainTimeDetection:
    def __init__(self, cfg_path, logger):
        """ 透過連續的剩餘量判斷出剩餘多少時間可以進食完畢
        Args:
            cfg_path: 子模塊配置文件
            logger: 保存運行過程中狀態的log實例化對象
        """
        from SpecialTopic.WorkingFlow.SubBlock.RemainTime.RemainTimeTransformerDetection \
            import RemainTimeTransformerDetection
        from SpecialTopic.WorkingFlow.SubBlock.RemainTime.RemainTimeTransformerDetectionTensorRT \
            import RemainTimeTransformerDetectionTensorRT
        from SpecialTopic.WorkingFlow.SubBlock.RemainTime.RemainTimeCalculate import RemainTimeCalculate
        from SpecialTopic.WorkingFlow.SubBlock.RemainTime.RemainTimeRegression import RemainTimeRegression
        from SpecialTopic.WorkingFlow.SubBlock.RemainTime.RemainTimeRegressionWithClasses \
            import RemainTimeRegressionWithClass
        from SpecialTopic.WorkingFlow.SubBlock.RemainTime.RemainTimeRegressionWithClassesTensorRT \
            import RemainTimeRegressionWithClassTensorRT
        from SpecialTopic.WorkingFlow.SubBlock.RemainTime.RemainTimeRegressionWithClassesForVerify \
            import RemainTimeRegressionWithClassForVerify
        from SpecialTopic.WorkingFlow.SubBlock.RemainTime.RemainTimeRegressionV2 import RemainTimeRegressionV2
        support_module = {
            'RemainTimeTransformerDetection': RemainTimeTransformerDetection,
            'RemainTimeTransformerDetectionTensorRT': RemainTimeTransformerDetectionTensorRT,
            'RemainTimeCalculate': RemainTimeCalculate,
            'RemainTimeRegression': RemainTimeRegression,
            'RemainTimeRegressionWithClass': RemainTimeRegressionWithClass,
            'RemainTimeRegressionWithClassTensorRT': RemainTimeRegressionWithClassTensorRT,
            'RemainTimeRegressionWithClassForVerify': RemainTimeRegressionWithClassForVerify,
            'RemainTimeRegressionV2': RemainTimeRegressionV2
        }
        cfg = parser_cfg(cfg_path)
        cfg_ = copy.deepcopy(cfg)
        self.cfg = cfg
        module_cls = get_cls_from_dict(support_module, cfg_)
        self.module = module_cls(**cfg_)
        # 將logger實例化對象放到module當中使用
        self.module.logger = logger
        module_name = cfg.get('type', None)
        logger['logger'].debug(f'RemainTimeDetection using [ {module_name} ] module')

    def __call__(self, call_api, inputs):
        results = self.module(call_api, inputs)
        return results
