# Working Flow Config

---
說明Config文件配置方式

### 主模塊必要參數
type = 說明要調用哪個主模塊\
config_file = 主模塊底下要使用哪個子模塊，json文件格式\
api = 要調用子模塊的哪些函數，如果需要順序調用多個就寫成list型態\
inputs = 該模塊輸入參數，如果一個子模塊當中需要調用多個函數就會需要是list型態\
outputs = 該模塊輸出參數，如果一個子模塊當中需要調用多個函數就會需要是list型態

### 子模塊必要參數
type = 說明要使用哪個子模塊

### working flow
整個工作流程，根據不同步驟會調用不同模塊\
會將每個模塊做成一個step，一個step表示一個工作模塊\
每個模塊會有指定的主模塊，底下的config_file就是主模塊下的子模塊，當中配置子模塊相關參數\
子模塊的配置需要到指定的json檔中撰寫\
每個模塊都會有input以及output，分別表示要吃入的參數以及輸出的參數\
注意: 上層step的output請盡量保持與下層step相同，如果直接使用**kwargs帶過會導致不好維護
如果覺得有缺少什麼參數可以直接修改並且發PR

### get_picture_cfg
獲取圖像資料的方式\
主要是從攝影機獲取圖像
##### api說明:
- get_single_picture: 獲取一張圖像資料
  - input: []
  - output: [image, track_object_info]
    - image = 獲取攝影機擷取的圖像
    - image_type = 影像的型態
- change_camera: 更換攝影機
  - input: [new_camera_id]
    - new_camera_id = 新攝影機的id
  - output: []

### object_detection_cfg
指定使用哪個目標檢測模型\
主要會將圖像進行目標檢測，同時會負責過濾短暫判別錯誤的標註對象，會給每個追蹤對象一個id方便接下來的模塊操作\
只要是同一個id表示是同一個目標
##### api說明:
- detect_single_picture: 對一張圖像進行預測，並返將圖像擷取出來以及該圖像屬於哪個id以及類別
  - input: [image, image_type, force_get_detect]
    - image = 圖像資料
    - image_type = 圖像資料的型態，目前支援[ndarray, Image]
    - force_get_detect = 獲取當前檢測內容(在測試時可以使用，這樣就可以獲取當前狀態)
  - output: [image, track_object_info, detect_results]
    - image = 傳入的圖像，這裡不會對傳入圖像做任何更動
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
    - detect_results = 當前目標檢測出的結果(要將force_get_detect開啟才會傳送，實際上線時不會使用)
##### 流程解釋:
透過指定模型進行目標檢測，將結果進行分析。目標對象會分成等待追蹤以及正在追蹤兩個類別
- 正在追蹤: 需要比較久的時間沒有檢測到目標才會被取消，新檢測到的目標會優先匹配到正在追蹤的目標，當一段時間有被追蹤到的比例夠時就會往下的模塊傳送
，一旦目標到正在追蹤就會給一個追蹤ID
- 等待追蹤: 無法匹配到正在追蹤的目標會先到等待追蹤，這部分只要短暫時間沒有再次偵測到就會被消除，等待追蹤的目標都不會往下個模塊進行傳遞
，被放到這裡的目標不會給予ID
這裡匹配目標的方式是透過計算交並比獲取，所以只需調整比例就可以寬鬆的認定為同一個目標

### object_classify_to_remain_classify
對照表，將目標檢測出的類別id轉換到剩餘量檢測的id，如此才可以調用正確的類別分類網路
##### api說明:
- get_remain_id: 獲取剩餘量對應的id

### remain_detection_cfg
剩餘量檢測模型設定
##### api說明:
- remain_detection: 對於剩餘量進行檢測

### show_results
將結果顯示出來
##### api說明:
- show_results: 將結果畫在圖上並且將圖像返回
