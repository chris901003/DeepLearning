import cv2


def main():
    image = cv2.imread(r'D:\Storage\61\ObjectDetection\imgs\6443d39a0bb20.jpg')
    width = image.shape[1]
    height = image.shape[0]
    with open(r'D:\Storage\61\ObjectDetection\annotations\6443d39a0bb20.txt', 'r') as f:
        box_info = f.readlines()
    box_info = box_info[0]
    box_info = box_info.split(" ")
    xmin, ymin, xmax, ymax = float(box_info[0]) * width, float(box_info[1]) * height, \
                             float(box_info[2]) * width, float(box_info[3]) * height
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
    show_img_open_cv(image)


def show_img_open_cv(img):
    cv2.namedWindow("My Image", 0)
    cv2.resizeWindow("My Image", 640, 960)  # 设置窗口大小
    # 顯示圖片，第一個參數表示視窗名稱，第二個參數就是你的圖片。
    cv2.imshow('My Image', img)

    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
