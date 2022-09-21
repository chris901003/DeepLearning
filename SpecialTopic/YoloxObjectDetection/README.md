# YoloxObjectDetection

使用yolox的目標檢測模型，使用模塊化方式構建模型，可以快速的替換不同模塊以及模塊中的超參數

### 推薦先備資料

1. 預訓練權重資料
   - [yolox_nano.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_nano.pth)
   - [yolox_tiny.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_tiny.pth)
   - [yolox_s.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_s.pth)
   - [yolox_m.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_m.pth)
   - [yolox_l.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_l.pth)
   - [yolox_x.pth](https://github.com/bubbliiiing/yolox-pytorch/releases/download/v1.0/yolox_x.pth)
2. 分類類別檔案(使用labelImg時就會自動產生)
3. [訓練的標註文件](https://github.com/chris901003/DeepLearning/blob/main/some_utils/labelImg2yolox.py)
4. [驗證的標註文件](https://github.com/chris901003/DeepLearning/blob/main/some_utils/labelImg2yolox.py)
5. [計算mAP需要的coco文件](https://github.com/chris901003/DeepLearning/blob/main/some_utils/labelImg2coco.py)

剩下模型可調的部分到train.py當中的parse_args注釋


### 其他資訊
訓練入口檔案: train.py\
驗證入口檔案: 未寫\
實際使用入口檔案: 未寫
