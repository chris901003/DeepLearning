from pydoc import cli
import socket
import numpy as np
import base64
import cv2
import time

bind_ip = '172.20.10.2'
bind_port = 8701

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((bind_ip, bind_port))
server.listen(5)
print(f"Serve is listing on ip:{bind_ip} , port:{bind_port}")
image = cv2.imread('/Users/huanghongyan/Documents/DeepLearning/mmsegmentation/data/ade/ADEChallengeData2016/images/tr'
                   'aining/ADE_train_00000001.jpg')
# ret, image = cv2.imencode('.jpg', image)
cap = cv2.VideoCapture(0)
count = 0
while True:
    client, addr = server.accept()
    print(f'Connect with:{addr}')
    while True:
        ret, image = cap.read()
        ret2, image = cv2.imencode('.jpg', image)
        client.send(image)
        print(count)
        count += 1
    client.close()
