import cv2 as cv
from my_utils.video_adapters import VideoWriterAdapter

class RGBDetector:
    def __init__(self, fire_handle, policy_gen):
        self.backSub = cv.bgsegm.createBackgroundSubtractorCNT()
        self.handle = fire_handle
        self.policy_gen = policy_gen

    def detect(self, read, step=-1) -> int:
        # print("rgb")
        policy = self.policy_gen(read.fps(), step)
        fid = read.frame_id()

        while step != 0:
            step -= 1
            frame = read()
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
