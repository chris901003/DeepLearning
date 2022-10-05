import argparse
import os
import torch
import cv2
import time
from SpecialTopic.ST.utils import get_classes
from SpecialTopic.YoloxObjectDetection.api import init_model as yolox_init_model
from SpecialTopic.YoloxObjectDetection.api import detect_image as object_detect_single_image
from SpecialTopic.ClassifyNet.api import init_model as remain_init_model
from SpecialTopic.ClassifyNet.api import detect_single_picture as remain_detect_single_image


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default=r'C:\Dataset\test_Trim.mp4')
    parser.add_argument('--object-detection-model-phi', type=str, default='l')
    parser.add_argument('--remain-detection-model-type', type=str, default='VIT')
    parser.add_argument('--remain-detection-model-phi', type=str, default='m')
    parser.add_argument('--object-detection-classes-path', type=str,
                        default=r'C:\Dataset\FoodDetectionDataset\classes.txt')
    parser.add_argument('--remain-detection-classes-path', type=str,
                        default=r'C:\Dataset\FoodLeft\classes.txt')
    parser.add_argument('--object-detection-pretrained', type=str,
                        default=r'C:\Checkpoint\YoloxFoodDetection\yolox_best_train_loss.pth')
    parser.add_argument('--remain-detection-pretrained', type=str,
                        default=r'C:\Checkpoint\RemainDetection\remain\remain_detection_x.pth')
    args = parser.parse_args()
    return args


def get_remain_detection_pretrained(num_classes, remain_pretrained):
    pretrained_path = list()
    for index in range(num_classes):
        fix_place = remain_pretrained.find('x')
        pretrained = remain_pretrained[:fix_place] + str(index) + '.pth'
        if os.path.exists(pretrained):
            pretrained_path.append(pretrained)
        else:
            pretrained_path.append('none')
    return pretrained_path


def get_remain_model(model_type, model_phi, remain_detection_pretrained, remain_classes, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = dict()
    for index in range(len(remain_detection_pretrained)):
        remain_model = remain_init_model(model_type=model_type, phi=model_phi, num_classes=remain_classes,
                                         pretrained=remain_detection_pretrained[index], device=device)
        results[str(index)] = remain_model
    return results


def remain_detection(remain_detection_model, remain_data, device):
    for data in remain_data:
        image = data['image']
        label = data['label']
        # from PIL import Image
        # img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # img.show()
        # img.save('./test.jpg')
        remain = remain_detect_single_image(model=remain_detection_model[str(label)], image=image, device=device)[0]
        remain = remain.softmax(dim=0)
        index = remain.argmax(dim=0).item()
        score = remain[index].item()
        data['remain'] = index
        data['remain_score'] = score
    return remain_data


def start_detection(video_path, device, object_detection_model, remain_detection_model, object_detection_num_classes,
                    object_detection_classes_name, remain_detection_classes_name):
    cap = cv2.VideoCapture(video_path)
    pTime = 0
    video_write = None
    while True:
        ret, frame = cap.read()
        if ret:
            img_height, img_width = frame.shape[:2]
            if video_write is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_name = './test.mp4'
                video_write = cv2.VideoWriter(video_name, fourcc, 30, (img_width, img_height))
            labels, scores, boxes = object_detect_single_image(object_detection_model, device, frame, [640, 640],
                                                               object_detection_num_classes)
            remain_data = list()
            index = 0
            for label, score, bbox in zip(labels, scores, boxes):
                bbox = [int(box) for box in bbox]
                ymin, xmin, ymax, xmax = bbox
                if ymin < 0 or xmin < 0 or ymax >= img_height or xmax >= img_width:
                    continue
                ymin, xmin, ymax, xmax = max(0, ymin), max(0, xmin), min(img_height, ymax), min(img_width, xmax)
                picture = frame[ymin:ymax, xmin:xmax, :]
                data = dict(image=picture, index=index, label=label, origin_position=(xmin, ymin, xmax, ymax),
                            score=score)
                index += 1
                remain_data.append(data)
            results = remain_detection(remain_detection_model, remain_data, device)
            for result in results:
                object_detection_label_index = result['label']
                origin_position = result['origin_position']
                remain_label_index = result['remain']
                object_detection_label = object_detection_classes_name[object_detection_label_index]
                remain_label = remain_detection_classes_name[remain_label_index]
                xmin, ymin, xmax, ymax = origin_position
                score = str(round(result['score'] * 100, 2))
                remain_score = str(round(result['remain_score'] * 100, 2))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                info = object_detection_label + '|' + score + '|' + remain_label + '|' + remain_score
                cv2.putText(frame, info, (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (89, 214, 210), 2, cv2.LINE_AA)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1080 // 2, 1920 // 2)
            cv2.imshow('img', frame)
            video_write.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = args_parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    video_path = args.video_path
    object_detection_model_phi = args.object_detection_model_phi
    remain_detection_model_type = args.remain_detection_model_type
    remain_detection_model_phi = args.remain_detection_model_phi
    object_detection_classes_path = args.object_detection_classes_path
    remain_detection_classes_path = args.remain_detection_classes_path
    object_detection_pretrained = args.object_detection_pretrained
    remain_detection_pretrained = args.remain_detection_pretrained
    object_detection_classes_name, object_detection_num_classes = get_classes(object_detection_classes_path)
    remain_detection_classes_name, remain_detection_num_classes = get_classes(remain_detection_classes_path)
    if not os.path.exists(object_detection_pretrained):
        object_detection_pretrained = 'none'
    remain_detection_pretrained = get_remain_detection_pretrained(object_detection_num_classes,
                                                                  remain_detection_pretrained)
    object_detection_model = yolox_init_model(pretrained=object_detection_pretrained, phi=object_detection_model_phi,
                                              num_classes=object_detection_num_classes, device=device)
    remain_detection_model = get_remain_model(remain_detection_model_type, remain_detection_model_phi,
                                              remain_detection_pretrained, remain_detection_num_classes, device)
    start_detection(video_path, device, object_detection_model, remain_detection_model, object_detection_num_classes,
                    object_detection_classes_name, remain_detection_classes_name)


if __name__ == '__main__':
    main()
    print('Finish')
