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
api說明:
- get_single_picture: 獲取一張圖像資料

### object_detection_cfg
指定使用哪個目標檢測模型\
api說明:
- detect_single_picture: 對一張圖像進行預測，並返將圖像擷取出來以及該圖像屬於哪個id以及類別

### object_classify_to_remain_classify
對照表，將目標檢測出的類別id轉換到剩餘量檢測的id，如此才可以調用正確的類別分類網路\
api說明:
- get_remain_id: 獲取剩餘量對應的id

### remain_detection_cfg
剩餘量檢測模型設定\
api說明:
- remain_detection: 對於剩餘量進行檢測

### show_results
將結果顯示出來\
api說明:
- show_results: 將結果畫在圖上並且將圖像返回
