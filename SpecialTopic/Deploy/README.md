# Deploy

### OnnxToTensorRT
將Onnx檔案格式的模型轉換成TensorRT引擎
- TensorRTBase = 給定一個最基礎的類，可以降低創建TensorRT引擎實例化的難度

### YoloxObjectDetection
專門將Yolox目標檢測模型，轉成Onnx的格式，這裡會盡可能追求效率的寫法，會將許多東西固定下來
