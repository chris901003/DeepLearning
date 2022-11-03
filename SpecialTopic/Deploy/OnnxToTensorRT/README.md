# Deploy: Onnx to TensorRT

### TensorrtBase
給一個最基礎的TensorRT的基底，可以直接透過該基底創建出一個TensorRT引擎，將引擎構建包裝起來
#### 初始化參數資料
- onnx_file_path = onnx檔案路徑位置
- fp16_mode = 是否使用fp16模式，預設會是fp32，使用fp16可以提升速度但同時會降低準確度
- max_batch_size = 最大batch，這裡主要是對固定batch的引擎做設定，如果batch部分是做成動態的就設定動態中的最大batch
- trt_engine_path = 如果有想要直接加載已經序列化好的TensorRT引擎就傳入資料位置
- save_trt_engine_path = 如果有想要保存TensorRT推理引擎就給一個保存位置
- dynamic_shapes = 保存要設定成動態的維度資料，以下是傳入的格式
```
dict = {'綁定到哪個輸入名稱': (最小輸入, 最常輸入, 最大輸入)}
        綁定到哪格輸入名稱 = 在構建onnx資料時會指定輸入的名稱，如果沒有自定義就會是系統給，查明後再填入
        輸入部分舉個例 = ((1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224))
        表示最小就會是1個batch，正常都會是4個batch，最多就只會有8個batch作為輸入
```
- dynamic_factor = 在使用動態輸入時會需要先預多開一些空間，如果發生空間不夠可以到這裡設定，正常來說保持1就可以
- max_workspace_size = 最大工作空間大小，如果遇到記憶體不足可到這裡改大
- trt_logger_level = TensorRT的Logger階級
- logger = 紀錄過程的logger
