import copy
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.ST.utils import get_cls_from_dict


class ReadPicture:
    def __init__(self, cfg_path, logger):
        """ 獲取圖像的主模塊
        Args:
            cfg_path: 設定子模塊的配置文件
            logger: 保存運行過程中狀態的log實例化對象
        """
        from SpecialTopic.WorkingFlow.SubBlock.ReadPicture.ReadPictureFromVideo import ReadPictureFromVideo
        from SpecialTopic.WorkingFlow.SubBlock.ReadPicture.ReadPictureFromCamera import ReadPictureFromCamera
        from SpecialTopic.WorkingFlow.SubBlock.ReadPicture.ReadPictureFromPi import ReadPictureFromPi
        from SpecialTopic.WorkingFlow.SubBlock.ReadPicture.ReadPictureFromKinectV2 import ReadPictureFromKinectV2
        from SpecialTopic.WorkingFlow.SubBlock.ReadPicture.ReadPictureFromVideoWithDepth import \
            ReadPictureFromVideoWithDepth
        from SpecialTopic.WorkingFlow.SubBlock.ReadPicture.ReadPictureFromD435 import ReadPictureFromD435
        support_module = {
            'ReadPictureFromVideo': ReadPictureFromVideo,
            'ReadPictureFromCamera': ReadPictureFromCamera,
            'ReadPictureFromPi': ReadPictureFromPi,
            'ReadPictureFromKinectV2': ReadPictureFromKinectV2,
            'ReadPictureFromVideoWithDepth': ReadPictureFromVideoWithDepth,
            'ReadPictureFromD435': ReadPictureFromD435
        }
        cfg = parser_cfg(cfg_path)
        self.cfg = cfg
        cfg_ = copy.deepcopy(cfg)
        module_cls = get_cls_from_dict(support_module, cfg_)
        self.module = module_cls(**cfg_)
        # 將log實例化對象放到module當中
        self.module.logger = logger
        module_name = cfg.get('type', None)
        logger['logger'].debug(f'ReadPicture using [ {module_name} ] module')

    def __call__(self, call_api, inputs):
        results = self.module(call_api, inputs)
        return results
