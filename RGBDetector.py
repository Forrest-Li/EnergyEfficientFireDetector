import numpy as np
import cv2 as cv
from my_utils.video_adapters import VideoWriterAdapter

class RGBDetector:
    def __init__(self, fire_handle, policy_gen):
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
                break

            fgMask = self.backSub.apply(frame)

            has_fire = self.handle.has_fire(frame, fgMask)
            if has_fire:
                print("has fire", fid)
            fire_frame = policy.update_frame_record(fid, has_fire)
            if fire_frame > -1:
                print("fire detected at frame", fire_frame)
                return fire_frame

            fid += 1

        return None

    def render(self, read):
        writer = VideoWriterAdapter(read)
        while True:
            frame = read()
            if frame is None:
                break

            fgMask = self.backSub.apply(frame)

            f = self.handle.apply_mask(frame.copy(), fgMask)
            writer(f)

        writer.release()
