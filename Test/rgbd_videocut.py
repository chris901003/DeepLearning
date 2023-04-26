import os
import cv2
import shutil
from pathlib import Path

path_root = Path(r"C:\DeepLearning\SpecialTopic\Verify\VideoSave")
folder_name = "Donburi16"
folder_path = path_root / (folder_name + "_cut")
os.makedirs(folder_path)
count = 0
cap_rgb = cv2.VideoCapture(str(path_root / folder_name / "RgbView.avi"))
cap_watch = cv2.VideoCapture(str(path_root / folder_name / "Stopwatch.mp4"))
cap_weight = cv2.VideoCapture(str(path_root / folder_name / "Weight.mp4"))
max_frame = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))


while True:
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, count)
    cap_watch.set(cv2.CAP_PROP_POS_FRAMES, count)
    cap_weight.set(cv2.CAP_PROP_POS_FRAMES, count)
    ret1, rgb = cap_rgb.read()
    ret2, watch = cap_watch.read()
    ret3, weight = cap_weight.read()
    if not (ret1 and ret2 and ret3):
        print("not read frame")
        exit()
    cv2.imshow("rgb", rgb)
    cv2.imshow("stopwatch", watch)
    cv2.imshow("weight", weight)
    print(count)
    K = cv2.waitKeyEx(0)
    if K == 2424832:
        count -= 1
        count = max(0, count)
    elif K == 2490368:
        count -= 10
        count = max(0, count)
    elif K == 2555904:
        count += 1
        count = min(max_frame, count)
    elif K == 2621440:
        count += 10
        count = min(max_frame, count)
    else:
        break
if count == 0:
    print("haven\'t cut anything")
    exit()
cv2.destroyAllWindows()

w = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
rgb_writer = cv2.VideoWriter(str(folder_path / "RgbView.avi"), cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h), 1)
w = int(cap_watch.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap_watch.get(cv2.CAP_PROP_FRAME_HEIGHT))
watch_writer = cv2.VideoWriter(str(folder_path / "Stopwatch.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h), 1)
w = int(cap_weight.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap_weight.get(cv2.CAP_PROP_FRAME_HEIGHT))
weight_writer = cv2.VideoWriter(str(folder_path / "Weight.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h), 1)
cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, count)
cap_watch.set(cv2.CAP_PROP_POS_FRAMES, count)
cap_weight.set(cv2.CAP_PROP_POS_FRAMES, count)

while True:
    ret1, rgb = cap_rgb.read()
    ret2, watch = cap_watch.read()
    ret3, weight = cap_weight.read()
    if not ((ret1 == ret2) and (ret2 == ret3)):
        print("video length not equal")
        print(ret1, " ", ret2, " ", ret3)
        exit()
    if not (ret1 and ret2 and ret3):
        print("video write over")
        break
    rgb_writer.write(rgb)
    watch_writer.write(watch)
    weight_writer.write(weight)
cap_rgb.release()
cap_watch.release()
cap_weight.release()
rgb_writer.release()
watch_writer.release()
weight_writer.release()

for file_fullname in os.listdir(path_root / folder_name):
    file_fullname = str(file_fullname)
    file_name = file_fullname.split(".", 1)
    if file_name[1] == "npy":
        file_name = file_name[0].split("_", 1)
        if file_name[0] != "Depth":
            print("warning! have some file type npy not name as \"Depth\"")
            print(file_name[0])
        nth_depth = int(file_name[1])
        if nth_depth < count:
            continue
        nth_depth -= count
        shutil.copyfile(str(path_root / folder_name / file_fullname), str(folder_path / ("Depth_" + str(nth_depth) + ".npy")))
print("depth write over")
