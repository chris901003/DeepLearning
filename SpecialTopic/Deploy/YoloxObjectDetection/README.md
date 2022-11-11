# Deploy YoloxObjectDetection

### Models
各種尺寸的模型轉onnx碼\
目前提供[Nano, Tiny, L]版本

#### YoloxObjectDetection_L
產生模型尺寸為L的onnx檔案，全部模型層結構都在檔案當中，這裡盡可能以提高效率為主的撰寫
#### YoloxObjectDetection_Nano
產生模型尺寸為Nano的onnx檔案，全部模型層結構都在檔案當中，這裡盡可能以提高效率為主的撰寫
#### YoloxObjectDetection_Tiny
產生模型尺寸為Tiny的onnx檔案，全部模型層結構都在檔案當中，這裡盡可能以提高效率為主的撰寫

### utils
提供該資料夾下的功能函數

### api
提供相關api直接使用，減少使用的負擔，可以快速使用各種接口

---
#### create_onnx_file
構建Yolox object detection的onnx模型格式
##### 輸入資料
- model_phi: 模型大小
- num_classes: 分類類別數量
- pretrained: 預訓練權重位置
- input_name: 輸入資料名稱
- output_name: 輸出資料名稱
- save_path: onnx檔案保存位置
- dynamic_axes: 動態維度資料
##### 輸出資料
- None，會直接將onnx檔案保存到指定位置

#### create_onnx_session
創建一個onnxruntime對象
##### 輸入資料
- onnx_file: onnx檔案路徑
##### 輸出資料
- 實例化的onnxruntime對象，可以執行的onnx

#### onnxruntime_detect_image
使用onnxruntime執行onnx格式的模型，這裡主要提供測試使用
##### 輸入資料
- onnx_model: onnx類型的模型 
- image: 圖像資料 
- input_shape: 輸入的圖像大小，如果是Default就會是[640, 640]
- num_classes: 類別數量 
- confidence: 置信度閾值 
- nms_iou: nms閾值 
- keep_ratio: 輸入圖像處理時是否需要保持圖像高寬比 
- input_name: 輸入到onnx模型的資料名稱，這裡需要與生成onnx資料時相同 
- output_name: 從onnx模型輸出的資料名稱，這裡需要與生成onnx資料時相同
##### 輸出資料
- top_label: 每個目標的類別 
- top_con: 置信度分數 
- top_boxes: 目標匡座標位置

#### create_tensorrt_engine
生成TensorRT推理引擎，包裝後的實例化對象
##### 輸入資料
- onnx_file_path: onnx檔案路徑 
- fp16_mode: 是否使用fp16模式，開啟後可以提升推理速度但是精準度會下降 
- max_batch_size: 最大batch資料，如果有使用動態batch這裡可以隨意填，如果是靜態batch就需要寫上 
- trt_engine_path: 如果已經有經過序列化保存的TensorRT引擎資料就直接提供，可以透過反序列化直接實例化對象 
- save_trt_engine_path: 如果有需要將TensorRT引擎序列化後保存，提供下次使用可以指定路徑 
- dynamic_shapes: 如果有設定動態輸入資料就需要到這裡指定
```python
min_shape, usual_shape, max_shape = (1, 3, 224, 224), (2, 3, 300, 300), (3, 3, 512, 512)
dict = {
    'input_name': (min_shape, usual_shape, max_shape)
}
# 如果有多組輸入都是變動shape就都放在dict當中
``` 
- trt_logger_level: TensorRT構建以及使用時的logger等級
##### 輸出資料
- TensorRT Engine包裝後的對象

#### tensorrt_engine_detect_image
TensorRT進行一次圖像檢測
##### 輸入資料
- tensorrt_engine: TensorRT引擎對象 
- image: 圖像資料 
- input_shape: 輸入到網路的圖像大小，Default = [640, 640]
- num_classes: 分類類別數 
- confidence: 置信度 
- nms_iou: nms閾值 
- keep_ratio: 原始圖像縮放時，是否需要保存高寬比 
- input_name: 輸入資料的名稱 
- output_shapes: 輸出資料的shape，主要是從tensorrt中的輸出都會是一維的，需要透過reshape還原
  - [shape1, shape2]
  有幾個輸出就需要提供多少種shape，這裡排放順序就會是輸出資料的順序，無法進行指定，所以要確定onnx建成時的輸出順序 
  如果使用Default就會默認Yolox預定的輸出shape 
- using_dynamic_shape: 如果有使用到動態shape這裡需要設定成True，否則無法告知引擎資料大小
##### 輸出資料
- top_label: 每個目標的類別 
- top_con: 置信度分數 
- top_boxes: 目標匡座標位置

### CameraDetection
使用相機進行檢測，可以查看使用TensorRT後的效果
