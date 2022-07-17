# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ChaseDB1Dataset(CustomDataset):
    """Chase_db1 dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_1stHO.png'.
    """

    # CLASSES = 這個數據及當中對應index的分類名稱，這裡是預設的如果有自定義的會優先使用自定義的
    CLASSES = ('background', 'vessel')

    # PALETTE = 對應的index在輸出的時候會用哪種顏色進行表示，這裡是預設的如果有自定義的會優先使用自定義的
    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, **kwargs):
        super(ChaseDB1Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_1stHO.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)
