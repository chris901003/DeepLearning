import os
import socket
import numpy as np
import cv2


class Webcam:
    def __init__(self, pi_ip="10.201.35.39", pi_port=10900, fps=10, w_h=[1280, 960]):
        if type(fps) != int:
            raise ValueError("fps 須為整數")
        resolution_set = [[640, 480], [800, 600], [1280, 720], [1280, 960], [1920, 1080]]
        if not (w_h in resolution_set):
            raise ValueError("解析度須為: [[640,480], [800,600], [1280,720], [1280,960], [1920,1080]]")
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((pi_ip, pi_port))
        except Exception:
            raise ValueError("連線錯誤")

        while True:
            try:
                self.client.send((str(fps)).encode())
                mess = str(w_h[0]) + " " + str(w_h[1])
                self.client.send(mess.encode())
                flag_pass = self.client.recv(1024).decode()
            except Exception:
                raise WindowsError("傳送接收失敗")
            if flag_pass != "ok":
                print(flag_pass)
                fps = input("fps: ")
                w_h = input("width and height")
            else:
                break
        self.w_h = w_h

    def get_img(self):
        try:
            self.client.send('get'.encode())
        except Exception:
            raise WindowsError("connection wrong when get img")

        byte_buffer = bytes()
        while byte_buffer.__len__() < (self.w_h[0] * self.w_h[1] * 3):
            byte_buffer += self.client.recv(1024*8)
        img = np.frombuffer(byte_buffer, dtype=np.uint8).reshape([self.w_h[1], self.w_h[0], 3])
        return img

    def __del__(self):
        self.client.send("end".encode())
        self.client.close()


test = Webcam()
os.system('pause')
wb_img = test.get_img()
print("here")

cv2.namedWindow('test', 0)
cv2.resizeWindow('test', 960, 1280)
cv2.imshow("test", wb_img)
cv2.waitKey()
