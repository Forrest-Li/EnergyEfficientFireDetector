import sys
ROOT = '/content/yolov5'
sys.path.append(ROOT)

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.augmentations import letterbox


class YOLODetector:
    @torch.no_grad()
    def __init__(self, frame_reader, frame_writer, weights, data, thres, line_thickness=3, imgsz=(640, 640)):
        self.read = frame_reader
        self.write = frame_writer
        self.frame_window = [0 for i in range(int(self.read.fps() / 2))]
        assert thres >= 0. and thres <= 1.
        self.thres = int(thres * len(self.frame_window))
        print(f"source fps: {self.read.fps()}, threshold: {self.thres} / {len(self.frame_window)}")
        self.device = select_device('')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=data, fp16=False)
        stride, pt = self.model.stride, self.model.pt
        self.imgsz = check_img_size(imgsz, s=stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        self.line_thickness = line_thickness
        self.dt = [0.0, 0.0, 0.0]

    @torch.no_grad()
    def inference(self, im):
        # for path, im, im0s, vid_cap, s in dataset:
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
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        self.dt[2] += time_sync() - t3

        return pred

    @torch.no_grad()
    def plot_prediction(self, pred, im0, im_shape):
        seen = 0
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.model.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im_shape[1:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{self.model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

    def frame_convert(self, img0):
        frame = letterbox(img0, self.imgsz, stride=self.model.stride, auto=self.model.pt)[0]
        # Convert
        frame = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        frame = np.ascontiguousarray(frame)
        return frame

    def detect(self) -> int:
        fid = 0
        len_frame_window = len(self.frame_window)
        num_fire_frames = 0
        self.read.reset()

        while True:
            img0 = self.read()
            if img0 is None:
                break

            frame = self.frame_convert(img0)
            fid_ = fid % len_frame_window
            num_fire_frames -= self.frame_window[fid_]
            has_fire = len(self.inference(frame)) > 0
            if has_fire:
                print("has fire:", fid)
            self.frame_window[fid_] = int(has_fire)
            num_fire_frames += self.frame_window[fid_]
            fid += 1

            if num_fire_frames >= self.thres:
                return fid

    def render(self):
        self.read.reset()
        fid = 0
        while True:
            print("frame:", fid)
            fid += 1
            img0 = self.read()
            if img0 is None:
                break
            frame = self.frame_convert(img0)
            pred = self.inference(frame)

            self.plot_prediction(pred, img0, frame.shape)

            self.write(img0)

        self.write.release()
