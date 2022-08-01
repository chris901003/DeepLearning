# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv

from mmocr.core import imshow_pred_boundary


class TextDetectorMixin:
    """Base class for text detector, only to show results.

    Args:
        show_score (bool): Whether to show text instance score.
    """

    def __init__(self, show_score):
        # 已看過
        # 保存show_score
        self.show_score = show_score

    def show_result(self,
                    img,
                    result,
                    score_thr=0.5,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results to draw over `img`.
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.imshow_pred_boundary`
        """
        # 已看過，將預測結果進行展示
        # img = 原始圖像讀入時的ndarray shape [height, width, channel]
        # result = 預測結果會是dict格式，其中會有boundary_result存放的會是標註的資料
        # score_thr = 置信度閾值，需要大於該值的才會顯示出來
        # bbox_color = 標註匡的顏色
        # text_color = 文字顏色
        # thickness = 粗細度
        # font_scale = 文字大小
        # win_name = 視窗名稱
        # wait_time = 等待時間
        # show = 是否直接展示圖像
        # out_file = 輸出圖像保存位置

        # 使用imread讀取img，這裡傳入的如果是ndarray型態就會直接返回
        img = mmcv.imread(img)
        # 將圖像拷貝一份
        img = img.copy()
        # 先將boundaries設定成None
        boundaries = None
        # 標籤部分設定成None
        labels = None
        if 'boundary_result' in result.keys():
            # 如果在result當中有boundary_result就會進來
            # 將boundaries部分取出來
            boundaries = result['boundary_result']
            # 將labels全部設定成0，因為我們只有一個類別就是文字類別
            labels = [0] * len(boundaries)

        # if out_file specified, do not show image in window
        if out_file is not None:
            # 如果有設定保存檔案位置就不會直接進行展示，將show設定成False
            show = False
        # draw bounding boxes
        if boundaries is not None:
            # 如果有標註匡就會進來，透過imshow_pred_boundary進行標注
            imshow_pred_boundary(
                img,
                boundaries,
                labels,
                score_thr=score_thr,
                boundary_color=bbox_color,
                text_color=text_color,
                thickness=thickness,
                font_scale=font_scale,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=out_file,
                show_score=self.show_score)

        if not (show or out_file):
            # 如果沒有設定out_file也沒有show就會跳出警告
            warnings.warn('show==False and out_file is not specified, '
                          'result image will be returned')
        # 回傳標註完的圖像
        return img
