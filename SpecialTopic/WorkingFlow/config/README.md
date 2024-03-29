# Working Flow Config

---
說明Config文件配置方式\
目前是根據主模塊分成不同資料夾，不過為了可以直接看所有文件的說明，這裡依舊會統一放在這裡說明該文件用法

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

## ReadPicture

---
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

### get_deep_picture_from_deepcamera
獲取圖像的方式，這裡主要會透過深度攝影機獲取圖像深度資料，同時也可以獨立從其他攝影機獲取rgb圖像
##### 參數說明:
- color_palette_type = 色彩條選擇
- color_palette_range = 色彩範圍，這裡需要根據使用的色彩條選擇範圍
- deep_image_height = 深度圖高度
- deep_image_width = 深度圖寬度
- rgb_camera = rgb圖像的攝影機，如果是Default就會是直接使用深度攝影機的rgb鏡頭\
 如果是要額外使用其他攝影機就需要提供對應ID，同時需要處理RGB圖像與深度圖像的關係
- rgb_image_height = 彩色圖高度
- rgb_image_width = 彩色圖寬度
- min_deep_value = 深度最小值
- max_deep_value = 深度最大值
- deep_match_rgb_cfg = 深度圖像映射到rgb圖像的方式，這裡傳入的需要是一個dict格式
##### api說明:
- get_single_picture: 獲取RGB圖像以及深度圖像資料
  - input: None
  - output: [rgb_image, image_type, deep_image, deep_draw]
    - rbg_image = 彩色圖像，這裡會是指定的相機獲取的RGB圖像
    - image_type = 圖像屬性類別
    - deep_image = 深度圖像資訊，這裡的高寬會與RGB圖像相同
    - deep_draw = 會依據不同深度映射到調色盤上的顏色進行著色

### get_deep_picture_from_kinectv2
獲取圖像的方式，這裡是專門給kinect v2使用的，如果使用不同深度攝影機就不適用
#### 參數說明
- deep_color_palette = 色彩條選擇
- deep_color_palette_range = 色彩範圍，這裡需要根據使用的色彩條選擇範圍
- deep_image_height = 深度圖高度
- deep_image_width = 深度圖寬度
- rgb_image_height = 彩色圖高度
- rgb_image_width = 彩色圖寬度
- deep_color_range = 深度顯示範圍
##### api說明:
- get_single_picture: 獲取RGB圖像以及深度圖像資料
  - input: None
  - output: [rgb_image, image_type, deep_image, deep_draw]
    - rbg_image = 彩色圖像，這裡會是指定的相機獲取的RGB圖像
    - image_type = 圖像屬性類別
    - deep_image = 深度圖像資訊，這裡的高寬會與RGB圖像相同
    - deep_draw = 會依據不同深度映射到調色盤上的顏色進行著色

### get_deep_picture_from_video
透過讀取彩色影片與深度影片獲取圖像資料，方便進行實驗，不過因為是直接保存深度資料，所以使用時請確保深度資料都是正確的
否則結果會有問題
#### 參數說明
- rgb_video_path = 彩色圖像檔案位置
- depth_video_path = 深度圖像檔案位置
- depth_view_video_path = 可視化深度圖像檔案位置
#### api說明
- get_single_picture: 獲取RGB圖像以及深度圖像資料
  - input: None
  - output: [rgb_image, image_type, deep_image, deep_draw]
    - rbg_image = 彩色圖像，這裡會是指定的相機獲取的RGB圖像
    - image_type = 圖像屬性類別
    - deep_image = 深度圖像資訊，這裡的高寬會與RGB圖像相同
    - deep_draw = 會依據不同深度映射到調色盤上的顏色進行著色
  - 注意事項: 這裡輸出的資料與不帶有深度的有所不同，所以在全流程的config中需要更改下層輸入的資料

## ObjectDetection

---
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
    - image = 傳入的圖像，這裡會將所有與圖像相關資料打包成dict，RGB圖像會在rgb_image當中 
      如果有深度資料就會有deep_image以及deep_draw資料
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
      - position = 位置資訊
      - category_from_object_detection = 分類類別名稱
      - object_score = 預測分數
      - track_id = 追蹤的ID
      - using_last = 是否需要進行之後層結構的判斷
    - detect_results = 當前目標檢測出的結果(要將force_get_detect開啟才會傳送，實際上線時不會使用)
##### 流程解釋:
透過指定模型進行目標檢測，將結果進行分析。目標對象會分成等待追蹤以及正在追蹤兩個類別
- 正在追蹤: 需要比較久的時間沒有檢測到目標才會被取消，新檢測到的目標會優先匹配到正在追蹤的目標，當一段時間有被追蹤到的比例夠時就會往下的模塊傳送
，一旦目標到正在追蹤就會給一個追蹤ID
- 等待追蹤: 無法匹配到正在追蹤的目標會先到等待追蹤，這部分只要短暫時間沒有再次偵測到就會被消除，等待追蹤的目標都不會往下個模塊進行傳遞
，被放到這裡的目標不會給予ID
這裡匹配目標的方式是透過計算交並比獲取，所以只需調整比例就可以寬鬆的認定為同一個目標

### object_detection_tensorrt_cfg
基本上與上方的object_detection_cfg相同，唯一不同的是將模型執行的底層從pytorch改成tensorrt\
經過測試後帧率大約提升20%，主要是記憶體消耗有明顯下降

## ObjectClassifyToRemainClassify

---
### object_classify_to_remain_classify
對照表，將目標檢測出的類別id轉換到剩餘量檢測的id，如此才可以調用正確的類別分類網路
##### api說明:
- get_remain_id: 獲取剩餘量對應的id
  - input: [image, track_object_info, using_dict_name]
    - image = 圖像資料，當前畫面的圖像
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
    - using_dict_name = 要使用哪個映射字典
  - output: [image, track_object_info]
    - image = 圖像資料，當前畫面的圖像
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
      - position = 位置資訊
      - category_from_object_detection = 分類類別名稱
      - object_score = 預測分數
      - track_id = 追蹤的ID
      - using_last = 是否需要進行之後層結構的判斷
      - remain_category_id = 在剩餘量檢測時使用到的模型ID(新增)

## RemainDetection

---
### remain_detection_cfg
剩餘量檢測模型設定
##### 參數說明:
- remain_module_file = 剩餘量模型設定，因為不同類別食物會使用不同剩餘量模型，所以需要寫成一個配置文件
  - 內部結構 = dict(dict)，表示那個類別的會使用哪個參數，第二個dict會指定模型大小以及預訓練權重位置
- classes_path = 剩餘量類別文件，如果有需要可以根據不同類別有不同剩餘量的類別檔案(目前還沒有寫)
- save_last_period = 一個追蹤ID的剩餘量資料可以保存多久，需要保存的目的是因為我們不會每幀都檢測一次剩餘量，因為意義不大反而增加負擔
- strict_down = 對於剩餘量的判斷是否要嚴格下降
##### api說明:
- remain_detection: 對於剩餘量進行檢測
  - input: [image, track_object_info]
    - image = 圖像資料，一定要是原始圖像，因為會對該圖像擷取需要的部分進行剩餘量判斷
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
  - output: [image, track_object_info]
    - image = 圖像資料，跟傳入時的圖像相同
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
      - position = 位置資訊
      - category_from_object_detection = 分類類別名稱
      - object_score = 預測分數
      - track_id = 追蹤的ID
      - using_last = 是否需要進行之後層結構的判斷
      - remain_category_id = 在剩餘量檢測時使用到的模型ID
      - category_from_remain = 剩餘量的類別(新增)

### remain_segformer_detection
使用分割網路將食物以及盤子進行分割，透過計算比例獲取剩餘量
##### 文件說明
- remain_module_file = 設定每種不同食物的模型大小以及權重
- classes_path = 類別標籤文件
- save_last_period = 一個index的圖像資訊會被保留多久
- with_color_platte = 調色盤選擇
- strict_down = 強制剩餘量只會往下降
- reduce_mode = 剩餘量模式(以下為選項)
  - momentum = 會參考上次的剩餘量來決定本次的剩餘量
    - alpha = 比例超參數
- area_mode = 計算面積的方式
  - area_mode = 強制碗需要包住食物
    - main_classes_idx = 主要對象在分割網路中的哪個類別index(食物)
    - sub_classes_idx = 次要對象在分割網路中的哪個類別index(碗)
  - pixel_mode = 直接計算每個類別有多少得像素就是有多少
    - main_classes_idx = 主要對象在分割網路中的哪個類別index(食物)
    - sub_classes_idx = 次要對象在分割網路中的哪個類別index(碗)
  - bbox_mode = 使用標註匡作為整個背景點醋量
    - main_classes_idx = 主要對象在分割網路中的哪個類別index(食物)
- check_init_ratio_frame = 有新目標要檢測剩餘量時需要以前多少幀作為100%的比例
  - 如果直接將(食物/(食物+碗))作為剩餘量判斷會有嚴重錯誤，因為可以知道這樣即使是滿的時候也不會是100%，
  所以這裡會是以前幾幀的佔比作為100%的標準，需要多幾幀是為了避免有誤測
- with_draw = 如果有需要獲取分割網路出來的色圖就設定為True，會在track_object_info當中多一個remain_color_picture
##### api說明
- remain_detection: 對於剩餘量進行檢測
  - input: [image, track_object_info]
    - image = 圖像資料，一定要是原始圖像，因為會對該圖像擷取需要的部分進行剩餘量判斷
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
  - output: [image, track_object_info]
    - image = 圖像資料，跟傳入時的圖像相同
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
      - position = 位置資訊
      - category_from_object_detection = 分類類別名稱
      - object_score = 預測分數
      - track_id = 追蹤的ID
      - using_last = 是否需要進行之後層結構的判斷
      - remain_category_id = 在剩餘量檢測時使用到的模型ID
      - category_from_remain = 剩餘量的類別，也有可能會是字串表示當前狀態(新增)
      - remain_color_picture = 分割網路預測結果的色圖，如果有開啟with_draw才會有

### remain_segformer_detection_with_deep_information
使用帶有深度資訊以及分割網路預測剩餘量百分比
##### 文件說明
- remain_module_file = 配置剩餘量模型的config資料，目前是根據不同類別會啟用不同的分割權重模型lasses_path: 分割網路的類別檔案 
- with_color_platte = 使用的調色盤 
- save_last_period = 最多可以保存多少幀沒有獲取到該id的圖像 
- strict_down = 強制剩餘量只會越來越少 
- reduce_mode = 對於當前檢測剩餘量的平均方式，可以減少預測結果突然暴增
  - momentum = 使用動量方式調整
    - alpha = 動量超參數
- area_mode = 計算當前剩餘量的方式 
  - volume_change = 透過體積比例變化得知剩餘量
- init_deep_mode = 初始化深度的方式，也就是定義原始底層深度的方式
  - target_seg_idx_man = 從指定類別中獲取基底深度值
    - target_seg_idx = 要從分割網路中選取哪些類別的值作為選取點
    - topk_freq = 估計基礎深度時，取出出現頻率最高的k個資料
- dynamic_init_deep_mode = 動態調整底層深度的方式，如果沒有打算動態調整就設定成None，或是初始化時不要傳入相關資料
  - 目前暫時棄用，沒有找到一個好的動態調整方式
- check_init_ratio_frame = 有新目標要檢測剩餘量時需要以前多少幀作為100%的比例 
- standard_remain_error = 在確認表準比例時的最大容忍標準差，這裡會是範圍表示可容忍的值建議[0.9, 1.1]
- with_seg_draw = 將分割的顏色標註圖回傳 
- with_depth_draw = 將深度圖像顏色回傳
##### api說明
- remain_detection: 對於剩餘量進行檢測
  - input: [image, track_object_info]
    - image = 圖像資料，一定要是原始圖像，因為會對該圖像擷取需要的部分進行剩餘量判斷
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
  - output: [image, track_object_info]
    - image = 圖像資料，跟傳入時的圖像相同
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
      - position = 位置資訊
      - category_from_object_detection = 分類類別名稱
      - object_score = 預測分數
      - track_id = 追蹤的ID
      - using_last = 是否需要進行之後層結構的判斷
      - remain_category_id = 在剩餘量檢測時使用到的模型ID
      - category_from_remain = 剩餘量的類別，也有可能會是字串表示當前狀態(新增)
      - remain_color_picture = 分割網路預測結果的色圖，如果有開啟with_seg_draw才會有
      - remain_deep_picture = 深度資料可視畫，如果有開啟with_depth_draw就會有

## ShowResults

---
### show_results
將結果顯示出來
##### 文件說明
- triangles: 打印矩形匡資料的都會在這裡
  - type = 說明給的座標資料型態，最後都會轉成[xmin, ymin, xmax, ymax]型態
  - val_name = 要從哪個Key獲取資料，只能從一個Key進行獲取，所以需要先打包好
  - color = 匡的顏色，不填寫會有默認值
  - thick = 匡的粗度，不填寫會有默認值
- texts: 文字相關資料都會在這裡
  - prefix = 字串開頭部分，如果不需要可以不要填或是用空
  - suffix = 字串結尾部份，如果不需要可以不要填或是用空
  - val_name = 將哪些資料寫到圖像上，這裡會盡可能的轉成str，如果無法轉換就會報錯
  - sep = 多個值之間的格開方式
  - color = 文字顏色，不填寫會有默認值
  - text_size = 文字大小，不填寫會有默認值
  - thick = 文字粗度，不填寫會有默認值
- pictures: 將圖像貼到指定位置上
  - val_name = 獲取圖像的變數名稱
  - position = 要貼到的座標位置
  - opacity = 透明度
##### api說明:
- show_results: 將結果畫在圖上並且將圖像返回
  - input: [image, track_object_info]
    - image = 圖像資料，跟傳入時的圖像相同
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
  - output: [image]
    - image = 標註好的圖像，如果有需要保存就可以直接保存

## RemainTimeDetection

---
### remain_time_transformer_detection
使用類似自然語言概念進行預測，這裡使用的會是基於transformer架構的nlp模型
##### 文件說明
- model_cfg = 模型配置
  - phi = 模型大小
  - setting_file_path = 模型當中配置參數，這個部分很重要，因為各種變數資料都會在這裡面，可以直接使用訓練時用的pickle檔
  - pretrained = 訓練好的權重路徑
- time_gap = 搜集多少時間的剩餘量檢測會放到剩餘時間檢測輸入
  - 主要的目的是在判斷剩餘時間時是透過固定時間檢測剩餘量判斷
- min_remain_detection = 同時可以要求最少要多少個剩餘量資料才會放到輸入當中
  - 避免檢測剩餘量資料太少會造成大幅波動，透過限制至少要多少剩餘量檢測才放入到剩餘時間檢測的輸入
- reduce_mode_buffer = 準備放入到剩餘時間檢測輸入時的資料率波方式
  - type = 指定方式，目前只有提供[mean, maximum, minimum, reduce_filter_maximum_and_minimum_mean]，如果有想到更好的可以提出
    - mean = 取平均值
    - maximum = 取最大值
    - minimum = 取最小值
    - reduce_filter_maximum_and_minimum_mean = 去除最大最小值後取平均
- reduce_mode_output = 輸出剩餘時間的率波方式
  - type = 指定方式，目前只有提供[momentum]，如果有想到更好的可以提出
    - momentum = 動量方式
      - alpha = 超參數，建議設定[0.5-0.7]左右
- keep_time = 一個追蹤對象在多久沒有追蹤到要拋棄
##### api說明
- remain_time_detection: 根據剩餘量與間隔時間檢測剩餘所需時間
  - input: [image, track_object_info]
    - image = 圖像資料，一定要是原始圖像，因為會對該圖像擷取需要的部分進行剩餘量判斷
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
  - output: [image, track_object_info]
    - image = 圖像資料，跟傳入時的圖像相同
    - track_object_info = 經過一系列操作後要傳到下個模塊的資料
      - position = 位置資訊
      - category_from_object_detection = 分類類別名稱
      - object_score = 預測分數
      - track_id = 追蹤的ID
      - using_last = 是否需要進行之後層結構的判斷
      - remain_category_id = 在剩餘量檢測時使用到的模型ID
      - category_from_remain = 剩餘量的類別，也有可能會是字串表示當前狀態(新增)
      - remain_color_picture = 分割網路預測結果的色圖，如果有開啟with_draw才會有
      - remain_time = 所需剩餘時間，會有可能跳出正在初始化(新)

## Log系統

---
在work_flow_cfg當中的key為log_config，所有的log設定都在此\
本log系統由巢狀構成，模塊依據主模塊進行分類
##### 文件說明
- log_link = 與哪個模塊做連接，連接名稱會是每個step的type部分
- level = 本層log的基礎level，注意這裡是會對log設定level不是handler
- format = 統一格式設定，如果該log的handler中的模塊沒有特別設定format就會直接套用這裡的format
- handler = 本log帶上的handler，會由list組成，可以有多個handler並且由type指定哪個handler
  - type = 指定的handler類別
  - save_path = 如果是FileHandler就會需要指定儲存檔案位置，其他的就不會需要設定
  - level = 專門給此handler使用的訊息等級
  - format = 專門給此handler使用的format方式，如果沒有設定就會用通用的
- sub_log = 本log下的子模塊，由list構成，裡面的每個設定都會是dict且型態就是一個新的log設定資料
