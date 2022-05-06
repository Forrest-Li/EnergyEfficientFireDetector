import time

from my_utils.video_adapters import CachedVideoCaptureAdapter, VideoWriterAdapter
from my_utils.FireHandle import FireHandle
from my_utils.filters import filter_jit, fire_filter
from my_utils.fire_policy import FirePolicy
from RGBDetector import RGBDetector
from YOLODetector import YOLODetector

'''
%lprun -f RGBDetector.detect RGBDetector(reader_fire, VideoWriterAdapter(reader_fire), \
                                         FireHandle(filter_jit, 0.00000001), 0.6).detect()
'''
def main():
    # Environment setup
    frame_info = {"start": -1, "end": -1, "count": -1}
    time_info = {"start": -1, "end": -1, "duration": -1}
    fps = -1
    reader_fire = CachedVideoCaptureAdapter("nest_2.mp4")
    detector = RGBDetector(reader_fire, VideoWriterAdapter(reader_fire),
                           FireHandle(fire_filter, 0.00000001), 0.6)
    # detector2 = YOLODetector(reader_fire, VideoWriterAdapter(reader_fire),
    #                          weights='yolov5s.pt', og_data=None, thres=0.5)

    # Detection starting
    detector.warmup()
    frame_info["start"] = 0
    time_info["start"] = time.time()
    frame_info["end"] = detector.render(reader_fire)  # TODO: remove this line
    # frame_info["end"] = detector.detect(frame_info["start"]) # TODO: un-comment this line
    time_info["end"] = time.time()
    # print(detector2.detect())
    detector.render()

    detected = frame_info["end"] != -1
    if detected:
        time_info["duration"] = (time_info["end"] - time_info["start"])
        frame_info["count"] = (frame_info["end"] - frame_info["start"])
        fps = (frame_info["count"]) / time_info["duration"]
    # Showing stats
    print("Detection information: (-1 := invalid)")
    print(f">Detected fire: {detected}")
    print(f">Detection taken: {time_info['duration']:.2f} sec")
    print(f">Frame count: {frame_info['count']} frames")
    print(f">FPS: {fps:.2f}")


if __name__ == '__main__':
    main()
