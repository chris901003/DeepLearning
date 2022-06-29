import cv2
import torch
from PIL import Image
from shells.deepsortor import Deepsortor
from shells.detector import Detector
from shells import tools


class Shell(object):
    def __init__(self, deepsort_config_path):
        self.deepsortor = Deepsortor(configFile=deepsort_config_path)
        self.detector = Detector()
        self.frameCounter = 0

    def update(self, im):
        retDict = {
            'frame': None,
            'list_of_ids': None,
            'obj_bboxes': []
        }

        self.frameCounter += 1
        frame = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        bboxes = self.detector.detect(frame)
        bbox_xywh = []
        confs = []

        if len(bboxes):
            for x1, y1, x2, y2, _, conf in bboxes:
                obj = [
                    int((x1 + x2) / 2), int((y1 + y2) / 2),
                    x2 - x1, y2 - y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)
            im, obj_bboxes = self.deepsortor.update(xywhs, confss, im)

            image = tools.plot_bboxes(im, obj_bboxes)
            retDict['frame'] = image
            retDict['obj_bboxes'] = obj_bboxes
        return retDict
