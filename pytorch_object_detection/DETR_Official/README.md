**DE⫶TR**: End-to-End Object Detection with Transformers
========

原始倉庫位置：https://github.com/facebookresearch/detr

#### 這裡主要是將程式碼加上中文註釋，基本上已經很清楚
訓練的指令可以到原作者官方網站查看，上面有連結

### 預測腳本使用
使用predict_screen進行預測，這裡會抓取螢幕畫面進行預測\
可以透過predict_screen中的frame = pyautogui.screenshot(region=[0, 0, 850, 500])中region部分選取要擷取的畫面位置\
如果要進行語意分割就在執行時加上--masks並且在detr.py的預訓練權重位置要給segmentation的權重檔案\
預測時需要的文檔位置都會在detr.py中，所以進行預測時需要更改檔案存放位置
