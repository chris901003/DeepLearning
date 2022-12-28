# Remain Eating Time Regression

主要目標: 將RemainEatingTimeRegression轉成Onnx以及TensorRT格式

### RemainEatingTimeRegression
主要模型位置，如果有需要對模型進行修改就從這裡改

### api
基本上所有接口都會在這裡，需要個別生成特定檔案都到這裡找

### PytorchToOnnxToTensorRT
就是簡單的將api當中的函數串起來，可以直接提供一串的服務，快速獲取需要的資料