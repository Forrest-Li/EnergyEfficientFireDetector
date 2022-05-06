from pathlib import Path
import numpy as np
import os
import sys

FILE = Path(__file__).resolve()
print(__file__)
ROOT = str(FILE.parents[0]) + '/yolov5' # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

import torch
from my_utils.video_adapters import VideoWriterAdapter


class YOLODetector:
    @torch.no_grad()
    def __init__(self, weights, data, policy_gen, line_thickness=3, imgsz=(640, 640)):
        self.policy_gen = policy_gen
        self.device = select_device('')
        self.model = DetectMultiBackend(
            weights, device=self.device, dnn=False, data=data, fp16=False)
        stride, pt = self.model.stride, self.model.pt
        self.imgsz = check_img_size(imgsz, s=stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        self.line_thickness = line_thickness
        self.dt = [0.0, 0.0, 0.0]

    @torch.no_grad()
    def inference(self, im):
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000
        classes = None
        agnostic_nms = False,  # class-agnostic NMS

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        self.dt[2] += time_sync() - t3

        return pred

    @torch.no_grad()
    def plot_prediction(self, pred, im0, im_shape):
        seen = 0
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            annotator = Annotator(
                im0, line_width=self.line_thickness, example=str(self.model.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im_shape[1:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{self.model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

    def frame_convert(self, img0):
        frame = letterbox(img0, self.imgsz,
                          stride=self.model.stride, auto=self.model.pt)[0]
        # Convert
        frame = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        frame = np.ascontiguousarray(frame)
        return frame

    def detect(self, read, step=-1) -> bool:
        # print("yolo")
        policy = self.policy_gen(read.fps(), step)
        fid = read.frame_id()

        while step != 0:
            step -= 1
            img0 = read()
            if img0 is None:
                return -1

            frame = self.frame_convert(img0)
            pred = self.inference(frame)
            has_fire = False
            for det in pred:
                if len(det):
                    has_fire = True
                    break
            if has_fire:
                print("yolo has fire", fid)
            fire_frame = policy.update_frame_record(fid, has_fire)
            if fire_frame > -1:
                print("fire detected at frame", fire_frame)
                return fire_frame

            fid += 1

        return None

    def render(self, read):
        writer = VideoWriterAdapter(read)
        fid = 0
        while True:
            img0 = read()
            if img0 is None:
                break
            print("frame: ", fid)
            fid += 1
            frame = self.frame_convert(img0)
            pred = self.inference(frame)

            self.plot_prediction(pred, img0, frame.shape)

            writer(img0)

        writer.release()
