import cv2
import os


def main():
    cap = cv2.VideoCapture(0)
    save_path = 'PoseVideo/3'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    assert os.path.isdir(save_path), '保存地址需要是資料夾路徑'
    cnt = len(os.listdir(save_path))
    imgs = list()
    recording = False
    while True:
        ret, img = cap.read()
        if ret:
            k = cv2.waitKey(1)
            if recording:
                imgs.append(img)
            if k == ord('q'):
                break
            elif k == ord('r'):
                if not recording:
                    recording = True
                else:
                    recording = False
                    photo_size = img.shape[:2][::-1]
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                    video_writer = cv2.VideoWriter(f'{save_path}/{cnt}.mp4', fourcc, 30, photo_size)
                    cnt += 1
                    for image in imgs:
                        video_writer.write(image)
                    imgs = list()
                    print('Finish write one video')
            state = 'Recording' if recording else 'Not recording'
            cv2.putText(img, state, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('img', img)


if __name__ == '__main__':
    main()
