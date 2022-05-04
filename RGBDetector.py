import numpy as np
import cv2 as cv


class RGBDetector:
    def __init__(self, frame_reader, frame_writer, fire_handle, thres):
        self.read = frame_reader
        self.write = frame_writer
        self.backSub = cv.bgsegm.createBackgroundSubtractorCNT()
        self.handle = fire_handle
        self.frame_window = [0 for i in range(int(self.read.fps() / 2))]
        assert thres >= 0. and thres <= 1.
        self.thres = int(thres * len(self.frame_window))
        print(f"source fps: {self.read.fps()}, threshold: {self.thres} / {len(self.frame_window)}")

    def warmup(self):
        frames = np.ones((50, 720, 1080, 3), dtype=np.uint8)
        for frame in frames:
            fgMask = self.backSub.apply(frame)
            has_fire = self.handle.has_fire(frame, fgMask)

    def detect(self) -> int:
        fid = 0
        len_frame_window = len(self.frame_window)
        num_fire_frames = 0
        self.read.reset()
        while True:
            frame = self.read()
            if frame is None:
                return -1

            fgMask = self.backSub.apply(frame)
            fid_ = fid % len_frame_window
            num_fire_frames -= self.frame_window[fid_]
            has_fire = self.handle.has_fire(frame, fgMask)
            if has_fire:
                print("has fire:", fid)
            self.frame_window[fid_] = int(has_fire)
            num_fire_frames += self.frame_window[fid_]

            fid += 1

            if num_fire_frames >= self.thres:
                return fid

    def render(self):
        self.read.reset()

        id = 0
        output_info = [] # ["currFrame", "areaRatio"]
        while True:
            frame = self.read()
            if frame is None:
                break

            fgMask = self.backSub.apply(frame)
            f, ratio = self.handle.apply_mask(frame.copy(), fgMask)
            self.write(f)

            output_info.append([id, ratio * 100])
            id += 1

        self.write.release()
        np.savetxt("plane.txt", np.array(output_info, dtype=np.uint), delimiter=",", fmt="%d")