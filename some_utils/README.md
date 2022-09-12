# 一些常用功能

### change_json_to_utf8
用來解決一些json編碼不符合規定的狀態

### labelme2coco
[原作者網址](https://www.cnblogs.com/gy77/p/15408027.html)
可以將從labelme標記好的東西轉換成coco格式

```
|-- images
|     |---  1.jpg
|     |---  1.json
|     |---  2.jpg
|     |---  2.json
|     |---  .......
|-- labelme2coco.py
|-- labels.txt
```

記得在labels中會需要
```
__ignore__
_background_
```

使用指令(記得要將labelme2coco.py放在同一個資料夾底下)
```commandline
 python labelme2coco.py --input_dir images --output_dir coco --labels labels.txt
```

### move_file
快速移動資料夾中的檔案

### heic_to_jpg
可以將iphone拍攝出來的HEIC格式的圖片批量轉成jpg格式

### labelImg2coco
獲取labelImg在yolo格式下標註的資料，輸出coco使用的annotation的json文件\
獲取annotation文件後還是需要自行將資料夾格式調整到coco樣式
