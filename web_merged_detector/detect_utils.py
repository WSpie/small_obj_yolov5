import logging
import random
import yaml
import time
import os
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

logger = logging.getLogger(__name__)


# only store ERROR while detecting and labeling
def logger_setting():
    logger.setLevel(logging.ERROR)
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger_file_handler = logging.FileHandler(os.path.join(log_dir, 'check.log'))
    logger.addHandler(logger_file_handler)
    logger_formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
    logger_file_handler.setFormatter(logger_formatter)


# box image
class ImgBox(object):
    def __init__(self, weight, img_sz, names, image=None, classes=None,
                 conf_thres=0.25, iou_thres=0.45, max_det=1000,
                 device='', agnostic_nms=False, augment=False):
        self.img = image
        self.names = names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.classes = classes
        self.device = select_device(device)
        self.model = attempt_load(weight, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_sz = check_img_size(img_sz, s=self.stride)

    def update(self, image):
        self.img = image

    def get_img_boxed(self):

        try:
            # load image, format: (w, h, c)
            img0 = self.img
            # resize and pad image while meeting stride-multiple constraints, format: (w, h, c)
            img = letterbox(img0, self.img_sz, stride=self.stride)[0]
            img0 = img
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # transpose image, format: (c, w, h)
            img = img[..., ::-1].transpose(2, 0, 1)
            # return a contiguous array in memory
            img = np.ascontiguousarray(img)

            # to tensor, format: (w, h, c)
            img = torch.from_numpy(img).to(self.device)
            # 0-255 to 0-1
            img = img.float()
            img /= 255.0
            # reformat image: (n, c, w, h)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # inference
            pred = self.model(img, augment=self.augment)[0]
            # if nothing detected, return None
            if not pred.tolist():
                return []

            # implement non max supression
            pred = non_max_suppression(pred, self.conf_thres,
                                       self.iou_thres,
                                       self.classes,
                                       self.agnostic_nms,
                                       max_det=self.max_det)
            box_attri = pred[0]
            # apply translation
            box_attri[:, :4] = scale_coords(img.shape[2:], box_attri[:, :4], img0.shape).round()
            # inference results format: [x1, y1, x2, y2, prob, class]
            box_attri = box_attri.tolist()
            for box_prop in box_attri:
                box_prop[-1] = self.names[int(box_prop[-1])]
            logger.info('[PEACE]')
        except:

            logger.exception('[OOPS]')

        else:
            return img0, box_attri


def draw_boxes(img, box_list, color_bar):
    if box_list != 0:
        for box in box_list:
            # draw boxes
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          color_bar[box[-1]], 1)
            # put and center the text
            text = f'{box[-1]}:{round(float(box[-2]), 1)}'
            text_length = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0][0]
            gap = (box[2] - box[0] - text_length) / 2
            cv2.putText(img, text, (int(box[0] + gap), int(box[1])), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=0.5, color=color_bar[box[-1]], thickness=1)
    return img


# main flow
def main(config):
    # initiate logger
    logger_setting()

    # load directory and class number
    with open(rf'{config}') as f:
        loader = yaml.load(f, Loader=yaml.FullLoader)
        root_dir = loader['data_path']
        class_num = loader['nc']
        weight = loader['weights']
        classes = loader['names']
        img_size = loader['img_size']
    # initiate colors
    colors = []
    for num in range(class_num):
        while True:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if not color in colors:
                colors.append(color)
                break
    color_bar = dict(zip(classes, colors))

    target_dir = root_dir + '/images'
    # start to infer images
    targets = os.listdir(target_dir)

    box_list = []

    target_box = ImgBox(weight=weight, img_sz=img_size, names=classes)

    for idx, target in enumerate(targets):

        target_path = target_dir + f'/{target}'
        if target.endswith(('.jpg', '.png', '.bmp', '.tiff')):
            start_time = time.time()
            target_img = cv2.imread(target_path)

            target_box.update(target_img)

            img_resized, box_info = target_box.get_img_boxed()
            box_list.append(box_info)
            painted_img = draw_boxes(img_resized, box_info, color_bar)
            target_save_dir = root_dir + '/results'
            if not os.path.exists(target_save_dir):
                os.mkdir(target_save_dir)
            cv2.imwrite(target_save_dir + f'/result_{target}', painted_img)
            end_time = time.time()
            print(f'Image processed, {end_time - start_time:.3f}s used!')


if __name__ == '__main__':
    config = './config/robot_car_data.yaml'
    # config = './config/color_tri_data.yaml'

    main(config)
