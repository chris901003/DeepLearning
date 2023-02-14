# EatingTime
剩餘時間驗證

### 使用說明
1. 使用SaveRgbdInfo進行錄影，會將錄影檔案存到指定的[folder-name]當中，內容會有一個Rgb的影片，還有許多的npy檔案，每一個幀會有一個檔案
也就是有多少幀就會有多少個npy檔案，這裡建議統一放在[C:\DeepLearning\SpecialTopic\Verify\EatingTime\RgbdSave]下
2. 使用RecordPredictTime對錄好的影片進行預測，會將結果存到指定的[ResultSavePath]當中，會是一個npy的檔案
   需要到[C:\DeepLearning\SpecialTopic\WorkingFlow\prepare\read_picture\rgbd_record_config.json]中指定影片的位置
3. 最後使用VerifyRemainTime進行損失計算，會從指定的[save-info-path]讀取上面產生的資料，並且會將最後結果存到指定的
   [result-save-root-folder\save-folder-name]下，會保存兩個檔案[chart.jpg, raw_info.json]，一個是圖表另一個是原始資料

### 其他檔案說明
1. CheckSaveRgbdInfo: 用來檢查錄影的結果是否正常
2. working_flow_cfg: 在使用WorkingFlow時會需要準備的設定檔
3. AveragePredictLoss: 將多個驗證資料整合做平均，可以了解在哪幾個部分的預測情況
   - 參數說明
      - num-part: 要將資料分成多少段，即使影片長度都不相同也會分成一樣的段數，只是每一段對上當時的時間長度會有所不同
      - save-folder: 每個影片驗證結果的根資料夾
      - select-infos: 要從原始驗證資料中獲取哪些資料
      - select-infos-title: 最後在畫圖表時，每個誤差的標題名稱
      - result-save-root-folder: 保存輸出的根目錄
      - result-folder-name: 根目錄下的哪個資料夾，最後資料會放到[result-save-root-folder\result-folder-name]資料夾下
   - 注意事項
     - 當使用不同select-infos時可能會需要添加取平均方式的函數，需要自行撰寫函數，並且添加到[support_avg_function]中的[support_func]
