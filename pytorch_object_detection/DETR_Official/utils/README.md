# Create coco annotation 使用說明
檔案擺放方式
需調整程式碼中的dataset_path到train或是val
```
├── my_yolo_dataset 自定义数据集根目录
│         ├── train   训练集目录
│         │     ├── images  训练集图像目录
│         │     ├── labels  训练集标签目录
|         |     └── categories.names 類別名稱
│         └── val    验证集目录
│               ├── images  验证集图像目录
│               ├── labels  验证集标签目录
|               └── categories.names 類別名稱
```            

labels裡面的bbox需要是(xmin, ymin, w, h)且為絕對座標的信息\
處理完後會在指定資料夾底下產生instances.json文件\
之後再按照coco數據集的擺放方式就可以了
