#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.image.misc import tensor2imgs
from mmcv.runner import load_checkpoint
from mmcv.utils.config import Config
from PIL import Image

try:
    import tesserocr
except ImportError:
    tesserocr = None

from mmocr.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector
from mmocr.models.textdet.detectors import TextDetectorMixin
from mmocr.models.textrecog.recognizer import BaseRecognizer
from mmocr.utils import is_type_list
from mmocr.utils.box_util import stitch_boxes_into_lines
from mmocr.utils.fileio import list_from_file
from mmocr.utils.model import revert_sync_batchnorm


# Parse CLI arguments
def parse_args():
    # 腳手架的一些參數
    parser = ArgumentParser()
    # 要預測的圖像檔案位置，也可以是資料夾會將其中的圖片讀出進行檢測
    parser.add_argument(
        'img', type=str, help='Input image file or folder path.')
    # 預測結果圖保存位置
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Output file/folder name for visualization')
    # 指定文字檢測模型(直接輸入模型名稱)，將文字部分匡選出來，不會進行判讀
    parser.add_argument(
        '--det',
        type=str,
        default='PANet_IC15',
        help='Pretrained text detection algorithm')
    # 指定文字檢測模型(模型配置config文件路徑)，將文字部分匡選出來，不會進行判讀
    parser.add_argument(
        '--det-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected det model. It '
        'overrides the settings in det')
    # 文字檢測模型訓練權重檔案路徑位置
    parser.add_argument(
        '--det-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected det model. '
        'It overrides the settings in det')
    # 文字內容判讀(直接輸入模型名稱)，放入的圖像需要是文字為主體，用來判讀圖像中文字內容
    parser.add_argument(
        '--recog',
        type=str,
        default='SEG',
        help='Pretrained text recognition algorithm')
    # 文字內容判讀(模型配置config文件路徑)，放入的圖像需要是文字為主體，用來判讀圖像中文字內容
    parser.add_argument(
        '--recog-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected recog model. It'
        'overrides the settings in recog')
    # 文字內容判讀訓練權重檔案路徑
    parser.add_argument(
        '--recog-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected recog model. '
        'It overrides the settings in recog')
    # 文字內容關鍵字(直接輸入模型名稱)，判斷文字當中的關鍵字部分
    parser.add_argument(
        '--kie',
        type=str,
        default='',
        help='Pretrained key information extraction algorithm')
    # 文字內容關鍵字(模型配置config文件路徑)，判斷文字當中的關鍵字部分
    parser.add_argument(
        '--kie-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected kie model. It'
        'overrides the settings in kie')
    # 文字內容關鍵字訓練權重檔案路徑
    parser.add_argument(
        '--kie-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected kie model. '
        'It overrides the settings in kie')
    # config檔案資料夾路徑，裡面會有所有的config文件
    parser.add_argument(
        '--config-dir',
        type=str,
        default=os.path.join(str(Path.cwd()), 'configs/'),
        help='Path to the config directory where all the config files '
        'are located. Defaults to "configs/"')
    # 在推理時是否使用batch_mode
    parser.add_argument(
        '--batch-mode',
        action='store_true',
        help='Whether use batch mode for inference')
    # 文字判斷時的batch_size
    parser.add_argument(
        '--recog-batch-size',
        type=int,
        default=0,
        help='Batch size for text recognition')
    # 文字偵測時的batch_size
    parser.add_argument(
        '--det-batch-size',
        type=int,
        default=0,
        help='Batch size for text detection')
    # 文字偵測以及文字判讀的batch_size
    parser.add_argument(
        '--single-batch-size',
        type=int,
        default=0,
        help='Batch size for separate det/recog inference')
    # 推理設備
    parser.add_argument(
        '--device', default=None, help='Device used for inference.')
    # 會將標註訊息寫成json檔案，用來指定json檔案要保存到哪裡
    parser.add_argument(
        '--export',
        type=str,
        default='',
        help='Folder where the results of each image are exported')
    # 輸出的格式，預設會是json格式
    parser.add_argument(
        '--export-format',
        type=str,
        default='json',
        help='Format of the exported result file(s)')
    # 是否需要保存係項資料
    parser.add_argument(
        '--details',
        action='store_true',
        help='Whether include the text boxes coordinates and confidence values'
    )
    # 是否需要將結果圖像用OpenCV展現
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    # 是否打印結果
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Prints the recognised text')
    # 是否要將相鄰的匡合併在一起
    parser.add_argument(
        '--merge', action='store_true', help='Merge neighboring boxes')
    # 多少的距離會將兩個匡進行合併，預設為20
    parser.add_argument(
        '--merge-xdist',
        type=float,
        default=20,
        help='The maximum x-axis distance to merge boxes')
    # 將參數打包起來
    args = parser.parse_args()
    if args.det == 'None':
        # 如果沒有特別設定分布式推理就會是None
        args.det = None
    if args.recog == 'None':
        # 如果沒有設定判斷文字的模型就會是None
        args.recog = None
    # Warnings
    if args.merge and not (args.det and args.recog):
        # 如果沒有同時使用文字偵測以及文字判別的話merge就會失效
        warnings.warn(
            'Box merging will not work if the script is not'
            ' running in detection + recognition mode.', UserWarning)
    if not os.path.samefile(args.config_dir, os.path.join(str(
            Path.cwd()))) and (args.det_config != ''
                               or args.recog_config != ''):
        # 就很機車，在debug時沒有調整當前路徑就會報錯
        warnings.warn(
            'config_dir will be overridden by det-config or recog-config.',
            UserWarning)
    # 回傳args參數內容
    return args


class MMOCR:

    def __init__(self,
                 det='PANet_IC15',
                 det_config='',
                 det_ckpt='',
                 recog='SEG',
                 recog_config='',
                 recog_ckpt='',
                 kie='',
                 kie_config='',
                 kie_ckpt='',
                 config_dir=os.path.join(str(Path.cwd()), 'configs/'),
                 device=None,
                 **kwargs):
        """ 已看過，MMOCR初始化部分
        Args:
            det: 文字檢測模型(直接輸入模型名稱)
            det_config: 文字檢測模型(輸入模型config文件)
            det_ckpt: 文字檢測模型訓練權重，輸入為檔案位置
            recog: 文字判讀模型(直接輸入模型名稱)
            recog_config: 文字判讀模型(輸入模型config文件)
            recog_ckpt: 文字判讀模型訓練權重，輸入為檔案位置
            kie: 關鍵字判讀模型(直接輸入模型名稱)
            kie_ckpt: 關鍵字判讀模型(輸入模型config文件)
            kie_config: 關鍵字判讀模型權重，輸入為檔案位置
            config_dir: 所有config文件根目錄
            device: 推理設備
        """

        # 文字匡選模型列表
        textdet_models = {
            # 以下就是有提供的文字匡選模型列表，如果使用指定模型名稱就會到這裡找到相關資訊
            'DB_r18': {
                # config文件位置
                'config':
                'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
                # 訓練權重地址
                'ckpt':
                'dbnet/'
                'dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
            },
            'DB_r50': {
                'config':
                'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth'
            },
            'DBPP_r50': {
                'config':
                'dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth'
            },
            'DRRG': {
                'config':
                'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'drrg/drrg_r50_fpn_unet_1200e_ctw1500_20211022-fb30b001.pth'
            },
            'FCE_IC15': {
                'config':
                'fcenet/fcenet_r50_fpn_1500e_icdar2015.py',
                'ckpt':
                'fcenet/fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth'
            },
            'FCE_CTW_DCNv2': {
                'config':
                'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py',
                'ckpt':
                'fcenet/' +
                'fcenet_r50dcnv2_fpn_1500e_ctw1500_20211022-e326d7ec.pth'
            },
            'MaskRCNN_CTW': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth'
            },
            'MaskRCNN_IC15': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
            },
            'MaskRCNN_IC17': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
            },
            'PANet_CTW': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
            },
            'PANet_IC15': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
            },
            'PS_CTW': {
                'config': 'psenet/psenet_r50_fpnf_600e_ctw1500.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
            },
            'PS_IC15': {
                'config':
                'psenet/psenet_r50_fpnf_600e_icdar2015.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth'
            },
            'TextSnake': {
                'config':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth'
            },
            'Tesseract': {}
        }

        # 文字判讀模型列表
        textrecog_models = {
            # 以下就是提供的模型名稱，可以透過名稱獲取對應的config文件以及訓練權重
            'CRNN': {
                'config': 'crnn/crnn_academic_dataset.py',
                'ckpt': 'crnn/crnn_academic-a723a1c5.pth'
            },
            'SAR': {
                'config': 'sar/sar_r31_parallel_decoder_academic.py',
                'ckpt': 'sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
            },
            'SAR_CN': {
                'config':
                'sar/sar_r31_parallel_decoder_chinese.py',
                'ckpt':
                'sar/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth'
            },
            'NRTR_1/16-1/8': {
                'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by16_1by8_academic_20211124-f60cebf4.pth'
            },
            'NRTR_1/8-1/4': {
                'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by8_1by4_academic_20211123-e1fdb322.pth'
            },
            'RobustScanner': {
                'config': 'robust_scanner/robustscanner_r31_academic.py',
                'ckpt': 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
            },
            'SATRN': {
                'config': 'satrn/satrn_academic.py',
                'ckpt': 'satrn/satrn_academic_20211009-cb8b1580.pth'
            },
            'SATRN_sm': {
                'config': 'satrn/satrn_small.py',
                'ckpt': 'satrn/satrn_small_20211009-2cf13355.pth'
            },
            'ABINet': {
                'config': 'abinet/abinet_academic.py',
                'ckpt': 'abinet/abinet_academic-f718abf6.pth'
            },
            'ABINet_Vision': {
                'config': 'abinet/abinet_vision_only_academic.py',
                'ckpt': 'abinet/abinet_vision_only_academic-e6b9ea89.pth'
            },
            'SEG': {
                'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
                'ckpt': 'seg/seg_r31_1by16_fpnocr_academic-72235b11.pth'
            },
            'CRNN_TPS': {
                'config': 'tps/crnn_tps_academic_dataset.py',
                'ckpt': 'tps/crnn_tps_academic_dataset_20210510-d221a905.pth'
            },
            'Tesseract': {},
            'MASTER': {
                'config': 'master/master_r31_12e_ST_MJ_SA.py',
                'ckpt': 'master/master_r31_12e_ST_MJ_SA-787edd36.pth'
            }
        }

        # 關鍵字判別模型
        kie_models = {
            'SDMGR': {
                'config': 'sdmgr/sdmgr_unet16_60e_wildreceipt.py',
                'ckpt':
                'sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
            }
        }

        # 保存指定文字匡選模型
        self.td = det
        # 保存指定文字判讀模型
        self.tr = recog
        # 保存指定關鍵字識別模型
        self.kie = kie
        # 保存運行設備
        self.device = device
        if self.device is None:
            # 如果沒有指定設備，就會檢查如果沒有gpu就會自動使用cpu進行推理
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        # Check if the det/recog model choice is valid
        if self.td and self.td not in textdet_models:
            # 如果有設定文字匡選模型，但是該模型沒有在模型列表當中就會報錯，所以如果有自行選擇模型可以將td設定成None
            raise ValueError(self.td,
                             'is not a supported text detection algorthm')
        elif self.tr and self.tr not in textrecog_models:
            # 如果有設定文字判讀模型，但是該模型沒有在模型列表當中就會報錯，所以如果有自行選擇模型可以將tr設定成None
            raise ValueError(self.tr,
                             'is not a supported text recognition algorithm')
        elif self.kie:
            # 與上方相同
            if self.kie not in kie_models:
                raise ValueError(
                    self.kie, 'is not a supported key information extraction'
                    ' algorithm')
            elif not (self.td and self.tr):
                # 如果要進行關鍵字提取需要將td與tr同時開啟才有用
                raise NotImplementedError(
                    self.kie, 'has to run together'
                    ' with text detection and recognition algorithms.')

        # 將detect_model先設定成None
        self.detect_model = None
        if self.td and self.td == 'Tesseract':
            # 如果使用的文字匡選模型是Tesseract就會檢查是否有安裝tesserocr模組，沒有安裝就會報錯
            if tesserocr is None:
                raise ImportError('Please install tesserocr first. '
                                  'Check out the installation guide at '
                                  'https://github.com/sirfz/tesserocr')
            # 將detect_model設定成Tesseract_det
            self.detect_model = 'Tesseract_det'
        elif self.td:
            # Build detection model，使用其他文字匡選模型就會到這裡
            if not det_config:
                # 如果沒有自行指定模型config就會到這裡，會獲取字典當中指定模型的config文件檔案位置
                det_config = os.path.join(config_dir, 'textdet/',
                                          textdet_models[self.td]['config'])
            if not det_ckpt:
                # 如果沒有指定訓練權重位置會透過網路下載字典當中指定的權重資料
                det_ckpt = 'https://download.openmmlab.com/mmocr/textdet/' + \
                    textdet_models[self.td]['ckpt']

            # 構建文字匡選模型，並且將權重載入進去
            self.detect_model = init_detector(
                det_config, det_ckpt, device=self.device)
            # 將模型當中Sync BN的部分換成BN，這樣才能在單gpu或是cpu上面進行推理
            self.detect_model = revert_sync_batchnorm(self.detect_model)

        self.recog_model = None
        if self.tr and self.tr == 'Tesseract':
            if tesserocr is None:
                raise ImportError('Please install tesserocr first. '
                                  'Check out the installation guide at '
                                  'https://github.com/sirfz/tesserocr')
            self.recog_model = 'Tesseract_recog'
        elif self.tr:
            # Build recognition model
            if not recog_config:
                recog_config = os.path.join(
                    config_dir, 'textrecog/',
                    textrecog_models[self.tr]['config'])
            if not recog_ckpt:
                recog_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'textrecog/' + textrecog_models[self.tr]['ckpt']

            # 構建文字判讀模型
            self.recog_model = init_detector(
                recog_config, recog_ckpt, device=self.device)
            # 將Sync BN換成BN，這樣才能在cpu或是單gpu下運行
            self.recog_model = revert_sync_batchnorm(self.recog_model)

        self.kie_model = None
        if self.kie:
            # Build key information extraction model
            if not kie_config:
                kie_config = os.path.join(config_dir, 'kie/',
                                          kie_models[self.kie]['config'])
            if not kie_ckpt:
                kie_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'kie/' + kie_models[self.kie]['ckpt']

            kie_cfg = Config.fromfile(kie_config)
            self.kie_model = build_detector(
                kie_cfg.model, test_cfg=kie_cfg.get('test_cfg'))
            self.kie_model = revert_sync_batchnorm(self.kie_model)
            self.kie_model.cfg = kie_cfg
            load_checkpoint(self.kie_model, kie_ckpt, map_location=self.device)

        # Attribute check，如果模型外面有多包一層為了平行處理的就需要剔除掉
        for model in list(filter(None, [self.recog_model, self.detect_model])):
            if hasattr(model, 'module'):
                model = model.module

    @staticmethod
    def get_tesserocr_api():
        """Get tesserocr api depending on different platform."""
        import subprocess
        import sys

        if sys.platform == 'linux':
            api = tesserocr.PyTessBaseAPI()
        elif sys.platform == 'win32':
            try:
                p = subprocess.Popen(
                    'where tesseract', stdout=subprocess.PIPE, shell=True)
                s = p.communicate()[0].decode('utf-8').split('\\')
                path = s[:-1] + ['tessdata']
                tessdata_path = '/'.join(path)
                api = tesserocr.PyTessBaseAPI(path=tessdata_path)
            except RuntimeError:
                raise RuntimeError(
                    'Please install tesseract first.\n Check out the'
                    ' installation guide at'
                    ' https://github.com/UB-Mannheim/tesseract/wiki')
        else:
            raise NotImplementedError
        return api

    def tesseract_det_inference(self, imgs, **kwargs):
        """Inference image(s) with the tesseract detector.

        Args:
            imgs (ndarray or list[ndarray]): image(s) to inference.

        Returns:
            result (dict): Predicted results.
        """
        is_batch = True
        if isinstance(imgs, np.ndarray):
            is_batch = False
            imgs = [imgs]
        assert is_type_list(imgs, np.ndarray)
        api = self.get_tesserocr_api()

        # Get detection result using tesseract
        results = []
        for img in imgs:
            image = Image.fromarray(img)
            api.SetImage(image)
            boxes = api.GetComponentImages(tesserocr.RIL.TEXTLINE, True)
            boundaries = []
            for _, box, _, _ in boxes:
                min_x = box['x']
                min_y = box['y']
                max_x = box['x'] + box['w']
                max_y = box['y'] + box['h']
                boundary = [
                    min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, 1.0
                ]
                boundaries.append(boundary)
            results.append({'boundary_result': boundaries})

        # close tesserocr api
        api.End()

        if not is_batch:
            return results[0]
        else:
            return results

    def tesseract_recog_inference(self, imgs, **kwargs):
        """Inference image(s) with the tesseract recognizer.

        Args:
            imgs (ndarray or list[ndarray]): image(s) to inference.

        Returns:
            result (dict): Predicted results.
        """
        is_batch = True
        if isinstance(imgs, np.ndarray):
            is_batch = False
            imgs = [imgs]
        assert is_type_list(imgs, np.ndarray)
        api = self.get_tesserocr_api()

        results = []
        for img in imgs:
            image = Image.fromarray(img)
            api.SetImage(image)
            api.SetRectangle(0, 0, img.shape[1], img.shape[0])
            # Remove beginning and trailing spaces from Tesseract
            text = api.GetUTF8Text().strip()
            conf = api.MeanTextConf() / 100
            results.append({'text': text, 'score': conf})

        # close tesserocr api
        api.End()

        if not is_batch:
            return results[0]
        else:
            return results

    def readtext(self,
                 img,
                 output=None,
                 details=False,
                 export=None,
                 export_format='json',
                 batch_mode=False,
                 recog_batch_size=0,
                 det_batch_size=0,
                 single_batch_size=0,
                 imshow=False,
                 print_result=False,
                 merge=False,
                 merge_xdist=20,
                 **kwargs):
        """ 已看過，首先會進入到這裡，如果是使用ocr.py
        Args:
            img: 被預測圖像檔案位置
            output: 預測結果圖輸出檔案位置
            details: 細節輸出檔案位置
            export: 總結文字檔案輸出位置
            export_format: 總結文字格式，預設為json
            batch_mode: 是否啟用batch模式
            recog_batch_size: 文字判別一個batch的大小
            det_batch_size: 文字匡選一個batch的大小
            single_batch_size: 同時設定文字匡選以及判斷的batch大小
            imshow: 是否需要將結果展現出來
            print_result: 打印出結果
            merge: 是否需要將鄰近的框框合併
            merge_xdist: 合併最大距離
        """

        # 透過locals會將當前區域變數進行返回，這裡會用copy進行拷貝
        args = locals().copy()
        # 將kwargs以及self部分進行剔除
        [args.pop(x, None) for x in ['kwargs', 'self']]
        # 將args裏面的內容使用Namespace進行包裝
        args = Namespace(**args)

        # Input and output arguments processing，將args進行加工
        # 主要是添加輸出預測圖像的檔案名稱以及細節輸出檔案名稱，以及將圖像進行讀入
        self._args_processing(args)
        self.args = args

        pp_result = None

        # Send args and models to the MMOCR model inference API
        # and call post-processing functions for the output
        if self.detect_model and self.recog_model:
            # 如果同時有設定文字匡選以及文字判讀模型就會到這裡
            # 會將文字匡選以及文字判讀模型放入同時也會有關鍵字提取的模型
            # det_recog_result = list[dict]格式，list長度就會是圖像數量，dict就會是預測結果資料
            det_recog_result = self.det_recog_kie_inference(
                self.detect_model, self.recog_model, kie_model=self.kie_model)
            # 透過det_recog_pp進行後處理
            pp_result = self.det_recog_pp(det_recog_result)
        else:
            # 其他情況，這裡就會使用文字匡選或是文字判讀其中一個
            for model in list(
                    filter(None, [self.recog_model, self.detect_model])):
                # 將模型以及圖像資料傳入到single_inference進行推理
                # result = list[dict]，list會是總共有多少張圖像，dict當中會有標註的訊息
                result = self.single_inference(model, args.arrays,
                                               args.batch_mode,
                                               args.single_batch_size)
                # 將模型以及預測結果輸入到single_pp當中，進行後處理
                pp_result = self.single_pp(result, model)

        return pp_result

    # Post processing function for end2end ocr
    def det_recog_pp(self, result):
        # 已看過，將文本匡選同時進行文本判讀部分合併的後處理部分
        # 最終結果保存的地方
        final_results = []
        # 獲取args設定
        args = self.args
        for arr, output, export, det_recog_result in zip(
                args.arrays, args.output, args.export, result):
            if output or args.imshow:
                if self.kie_model:
                    res_img = det_recog_show_result(arr, det_recog_result)
                else:
                    res_img = det_recog_show_result(
                        arr, det_recog_result, out_file=output)
                if args.imshow and not self.kie_model:
                    mmcv.imshow(res_img, 'inference results')
            if not args.details:
                simple_res = {}
                simple_res['filename'] = det_recog_result['filename']
                simple_res['text'] = [
                    x['text'] for x in det_recog_result['result']
                ]
                final_result = simple_res
            else:
                final_result = det_recog_result
            if export:
                mmcv.dump(final_result, export, indent=4)
            if args.print_result:
                print(final_result, end='\n\n')
            final_results.append(final_result)
        return final_results

    # Post processing function for separate det/recog inference
    def single_pp(self, result, model):
        # 已看過，後處理部分
        # result = 預測結果
        # model = 模型本身

        # 將原始圖像以及資料輸出位置以及詳細資料輸出位置以及預測結果進行遍歷
        for arr, output, export, res in zip(self.args.arrays, self.args.output,
                                            self.args.export, result):
            if export:
                # 如果有設定詳細資訊輸出位置就會到這裡，將res的資料透過dump寫入
                mmcv.dump(res, export, indent=4)
            if output or self.args.imshow:
                # 如果有設定將結果show出來或是需要output到資料夾就會到這裡
                if model == 'Tesseract_det':
                    # 如果使用的是Tesseract會到這裡特別處理
                    res_img = TextDetectorMixin(show_score=False).show_result(
                        arr, res, out_file=output)
                elif model == 'Tesseract_recog':
                    # 如果使用的是Tesseract會到這裡特別處理
                    res_img = BaseRecognizer.show_result(
                        arr, res, out_file=output)
                else:
                    # 其他模型會到這裡，res_img = 經過標註後的圖像
                    res_img = model.show_result(arr, res, out_file=output)
                if self.args.imshow:
                    # 如果需要直接展示就會到這裡使用imshow函數
                    mmcv.imshow(res_img, 'inference results')
            if self.args.print_result:
                # 最後打印結果
                print(res, end='\n\n')
        # 回傳result
        return result

    def generate_kie_labels(self, result, boxes, class_list):
        idx_to_cls = {}
        if class_list is not None:
            for line in list_from_file(class_list):
                class_idx, class_label = line.strip().split()
                idx_to_cls[class_idx] = class_label

        max_value, max_idx = torch.max(result['nodes'].detach().cpu(), -1)
        node_pred_label = max_idx.numpy().tolist()
        node_pred_score = max_value.numpy().tolist()
        labels = []
        for i in range(len(boxes)):
            pred_label = str(node_pred_label[i])
            if pred_label in idx_to_cls:
                pred_label = idx_to_cls[pred_label]
            pred_score = node_pred_score[i]
            labels.append((pred_label, pred_score))
        return labels

    def visualize_kie_output(self,
                             model,
                             data,
                             result,
                             out_file=None,
                             show=False):
        """Visualizes KIE output."""
        img_tensor = data['img'].data
        img_meta = data['img_metas'].data
        gt_bboxes = data['gt_bboxes'].data.numpy().tolist()
        if img_tensor.dtype == torch.uint8:
            # The img tensor is the raw input not being normalized
            # (For SDMGR non-visual)
            img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        else:
            img = tensor2imgs(
                img_tensor.unsqueeze(0), **img_meta.get('img_norm_cfg', {}))[0]
        h, w, _ = img_meta.get('img_shape', img.shape)
        img_show = img[:h, :w, :]
        model.show_result(
            img_show, result, gt_bboxes, show=show, out_file=out_file)

    # End2end ocr inference pipeline
    def det_recog_kie_inference(self, det_model, recog_model, kie_model=None):
        """ 已看過，端到端的文本檢測，會先將文字匡選出來之後進行文字判讀，如果有需要可以進行關鍵字提取
        Args:
            det_model: 文本匡選模型
            recog_model: 文本判讀模型
            kie_model: 文本關鍵字提取模型
        """
        # 最終結果保存的地方
        end2end_res = []
        # Find bounding boxes in the images (text detection)
        # 獲取當前圖像的文字標註匡範圍，這裡就會是匡選的結果資料
        det_result = self.single_inference(det_model, self.args.arrays,
                                           self.args.batch_mode,
                                           self.args.det_batch_size)
        # 將匡選資料提取出來，bboxes_list = list[list[list]]，第一個list是batch_size，第二個list該圖像有多少文字匡，第三個list是座標
        bboxes_list = [res['boundary_result'] for res in det_result]

        if kie_model:
            # 如果有關鍵文字偵測就會到這裡
            kie_dataset = KIEDataset(
                dict_file=kie_model.cfg.data.test.dict_file)

        # For each bounding box, the image is cropped and
        # sent to the recognition model either one by one
        # or all together depending on the batch_mode
        # 這裡會遍歷(原始圖像檔案名稱, 圖像ndarray, 文本匡選座標, 標註後圖像輸出的檔案路徑)
        for filename, arr, bboxes, out_file in zip(self.args.filenames,
                                                   self.args.arrays,
                                                   bboxes_list,
                                                   self.args.output):
            # 保存資料的部分
            img_e2e_res = {}
            img_e2e_res['filename'] = filename
            img_e2e_res['result'] = []
            # 保存文本匡資訊
            box_imgs = []
            for bbox in bboxes:
                box_res = {}
                # 這裡會將座標部分拿出來
                box_res['box'] = [round(x) for x in bbox[:-1]]
                # bbox當中最後一個值是匡選區域平均置信度分數
                box_res['box_score'] = float(bbox[-1])
                # 如果輸出的標註方式是矩形就只會有8個數值，這裡就直接取出來
                box = bbox[:8]
                if len(bbox) > 9:
                    # 如果輸出的使多邊形就會超過8個值
                    # 這裡會找到所有點當中的(xmin, ymin, xmax, ymax)
                    min_x = min(bbox[0:-1:2])
                    min_y = min(bbox[1:-1:2])
                    max_x = max(bbox[0:-1:2])
                    max_y = max(bbox[1:-1:2])
                    # 最終更新box資訊
                    box = [
                        min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                    ]
                # 將指定地方的圖像進行裁切
                box_img = crop_img(arr, box)
                if self.args.batch_mode:
                    # 如果有設定batch模式就會先保存起來，最後再一次進行推理
                    box_imgs.append(box_img)
                else:
                    # 沒有設定batch模式就會到這裡，每次只會預測一個標註匡
                    if recog_model == 'Tesseract_recog':
                        # 使用Tesseract模型就會到這裡
                        recog_result = self.single_inference(
                            recog_model, box_img, batch_mode=True)
                    else:
                        # 其他判讀模型就會到這裡
                        recog_result = model_inference(recog_model, box_img)
                    text = recog_result['text']
                    text_score = recog_result['score']
                    if isinstance(text_score, list):
                        text_score = sum(text_score) / max(1, len(text))
                    box_res['text'] = text
                    box_res['text_score'] = text_score
                # 將最後的預測結果放到img_e2e_res當中
                img_e2e_res['result'].append(box_res)

            if self.args.batch_mode:
                # 如果有設定batch模式就會到這裡，一次將裁切出來的部分進行文字判讀
                recog_results = self.single_inference(
                    recog_model, box_imgs, True, self.args.recog_batch_size)
                for i, recog_result in enumerate(recog_results):
                    text = recog_result['text']
                    text_score = recog_result['score']
                    if isinstance(text_score, (list, tuple)):
                        text_score = sum(text_score) / max(1, len(text))
                    img_e2e_res['result'][i]['text'] = text
                    img_e2e_res['result'][i]['text_score'] = text_score

            if self.args.merge:
                # 如果有設定merge就會到這裡
                img_e2e_res['result'] = stitch_boxes_into_lines(
                    img_e2e_res['result'], self.args.merge_xdist, 0.5)

            if kie_model:
                # 如果有需要進行關鍵字檢測就會到這裡
                annotations = copy.deepcopy(img_e2e_res['result'])
                # Customized for kie_dataset, which
                # assumes that boxes are represented by only 4 points
                for i, ann in enumerate(annotations):
                    min_x = min(ann['box'][::2])
                    min_y = min(ann['box'][1::2])
                    max_x = max(ann['box'][::2])
                    max_y = max(ann['box'][1::2])
                    annotations[i]['box'] = [
                        min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                    ]
                ann_info = kie_dataset._parse_anno_info(annotations)
                ann_info['ori_bboxes'] = ann_info.get('ori_bboxes',
                                                      ann_info['bboxes'])
                ann_info['gt_bboxes'] = ann_info.get('gt_bboxes',
                                                     ann_info['bboxes'])
                kie_result, data = model_inference(
                    kie_model,
                    arr,
                    ann=ann_info,
                    return_data=True,
                    batch_mode=self.args.batch_mode)
                # visualize KIE results
                self.visualize_kie_output(
                    kie_model,
                    data,
                    kie_result,
                    out_file=out_file,
                    show=self.args.imshow)
                gt_bboxes = data['gt_bboxes'].data.numpy().tolist()
                labels = self.generate_kie_labels(kie_result, gt_bboxes,
                                                  kie_model.class_list)
                for i in range(len(gt_bboxes)):
                    img_e2e_res['result'][i]['label'] = labels[i][0]
                    img_e2e_res['result'][i]['label_score'] = labels[i][1]

            end2end_res.append(img_e2e_res)
        return end2end_res

    # Separate det/recog inference pipeline
    def single_inference(self, model, arrays, batch_mode, batch_size=0):
        """ 已看過，將文字匡選以及文字判讀進行分開
        Args:
            model: 模型本身，可能會是文字匡選模型或是文字判讀模型
            arrays: list[ndarray]，圖像的ndarray shape [height, width, channel]，且通道是BGR排序，list長度就會是有多少張圖像
            batch_mode: 是否啟用batch模式
            batch_size: 一次傳入多少張圖像
        """

        def inference(m, a, **kwargs):
            """ 已看過
            Args:
                m: 模型本身
                a: 輸入的圖像
                kwargs: 說明是否啟用batch模式
            """
            if model == 'Tesseract_det':
                # 如果使用的是Tesseract的文字匡選模型就會到這裡
                return self.tesseract_det_inference(a)
            elif model == 'Tesseract_recog':
                # 如果是用的是Tesseract的文字判讀模型就會到這裡
                return self.tesseract_recog_inference(a)
            else:
                # 其他模型就會到這裡
                return model_inference(m, a, **kwargs)

        # 最終回傳結果的list
        result = []
        if batch_mode:
            # 如果有啟用batch模式就會到這裡
            if batch_size == 0:
                # 如果沒有指定batch_size就直接將整個arrays放入
                # 將結果放到result當中
                result = inference(model, arrays, batch_mode=True)
            else:
                # 如果有指定batch_size
                n = batch_size
                # 將arrays依據batch_size進行分割
                arr_chunks = [
                    arrays[i:i + n] for i in range(0, len(arrays), n)
                ]
                # 遍歷分割後的圖像
                for chunk in arr_chunks:
                    # 將結果放到result當中
                    result.extend(inference(model, chunk, batch_mode=True))
        else:
            # 如果沒有設定batch模式，就會嚴格遍歷每一張圖
            for arr in arrays:
                # 將結果放到result當中
                result.append(inference(model, arr, batch_mode=False))
        # 最終將result進行回傳，裏面至少會有標註訊息list[list]型態key值為boundary_result
        return result

    # Arguments pre-processing function
    def _args_processing(self, args):
        # 已看過，傳入的args會是Namespace實例對象，裡面會有設定的參數
        # Check if the input is a list/tuple that
        # contains only np arrays or strings
        if isinstance(args.img, (list, tuple)):
            # 如果傳入的img是list或是tuple型態就會到這裡
            img_list = args.img
            # 檢查存的資料要是圖像檔案位置或是已經讀取出來的ndarray型態
            if not all([isinstance(x, (np.ndarray, str)) for x in args.img]):
                raise AssertionError('Images must be strings or numpy arrays')

        # Create a list of the images
        if isinstance(args.img, str):
            # 如果傳入的是圖像檔案位置會到這裡
            img_path = Path(args.img)
            if img_path.is_dir():
                # 如果傳入的是資料夾路徑，就會將當中的圖像檔案名稱讀取出來
                img_list = [str(x) for x in img_path.glob('*')]
            else:
                # 否則就直接用list包裝起來
                img_list = [str(img_path)]
        elif isinstance(args.img, np.ndarray):
            # 如果是已經轉成ndarray的圖像就直接使用
            img_list = [args.img]

        # Read all image(s) in advance to reduce wasted time
        # re-reading the images for visualization output
        # 透過imread將圖像讀取出來，這裡會是ndarray型態shape [height, width, channel]，且為BGR排列
        args.arrays = [mmcv.imread(x) for x in img_list]

        # Create a list of filenames (used for output images and result files)
        if isinstance(img_list[0], str):
            # 如果img_list裡面存的是圖像路徑就會到這裡，將檔案名稱進行保存
            args.filenames = [str(Path(x).stem) for x in img_list]
        else:
            # 如果傳入的是ndarray我們就只會給index編號
            args.filenames = [str(x) for x in range(len(img_list))]

        # If given an output argument, create a list of output image filenames
        # 獲取總共會有多少張圖像
        num_res = len(img_list)
        if args.output:
            # 如果有指定預測結果圖像保存位置就會到這裡
            output_path = Path(args.output)
            if output_path.is_dir():
                # 如果給定的保存位置是資料夾檔案位置就會到這裡產生最終檔案名稱
                args.output = [
                    str(output_path / f'out_{x}.png') for x in args.filenames
                ]
            else:
                # 否則就直接會是指定的檔案名稱，這裡包含副檔名
                args.output = [str(args.output)]
                if args.batch_mode:
                    # 如果有啟用batch_mode會輸出多張圖像，需要傳入的是資料夾檔案位置
                    raise AssertionError('Output of multiple images inference'
                                         ' must be a directory')
        else:
            # 如果沒有指定output位置就全部都是None
            args.output = [None] * num_res

        # If given an export argument, create a list of
        # result filenames for each image
        if args.export:
            # 如果有設定輸出詳細資料的檔案保存位置就會進來
            export_path = Path(args.export)
            # 這裡也是產生檔案名稱
            args.export = [
                str(export_path / f'out_{x}.{args.export_format}')
                for x in args.filenames
            ]
        else:
            # 如果沒有設定就會是None
            args.export = [None] * num_res

        # 回傳處理好的args
        return args


# Create an inference pipeline with parsed arguments
def main():
    # 獲取args參數資料
    args = parse_args()
    # 實例化MMOCR
    ocr = MMOCR(**vars(args))
    # 使用readtext函數，對輸入的圖像進行偵測
    ocr.readtext(**vars(args))


if __name__ == '__main__':
    # 呼叫主函數
    main()
