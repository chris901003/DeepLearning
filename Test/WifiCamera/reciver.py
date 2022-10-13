import base64
import socket
import cv2
import numpy as np
import json
import time

host = '10.201.14.172'
port = 8700

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))

last_img = None
while True:
    cmd = 'hello world'
    while True:
        data = s.recv(1048576)
        buffer = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if img is not None:
            last_img = img
        cv2.imshow('img', last_img)
        time.sleep(0.01)
        if cv2.waitKey(1) == ord('q'):
            break
    # print(f'Server return : {data}')
