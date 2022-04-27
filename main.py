import time
from YOLODetector import YOLODetector
# import numpy as np
# import cv2
# from imutils.video import FPS
# from tqdm import tqdm
from my_utils.video_adapters import CachedVideoCaptureAdapter, VideoWriterAdapter
from my_utils.FireHandle import FireHandle
from my_utils.filters import filter_jit
from my_utils.fire_policy import FirePolicy
from RGBDetector import RGBDetector

'''
%lprun -f RGBDetector.detect RGBDetector(reader_fire, VideoWriterAdapter(reader_fire), \
                                         FireHandle(filter_jit, 0.00000001), 0.6).detect()
'''


def main():
    # Environment setup
    frame_info = {"start": -1, "end": -1, "count": -1}
    time_info = {"start": -1, "end": -1, "duration": -1}
    fps = -1
    reader_fire = CachedVideoCaptureAdapter("fire.mp4")
    # detector = RGBDetector(FireHandle(filter_jit, 0.01), FirePolicy.make_policy(1, 0.6))
    detector = YOLODetector(
        "./weights/best.pt", None, FirePolicy.make_policy(1, 0.6), 1)
    # Detection starting
    frame_info["start"] = 0
    time_info["start"] = time.time()
    frame_info["end"] = detector.render(reader_fire)  # TODO: remove this line
    # frame_info["end"] = detector.detect(frame_info["start"]) # TODO: un-comment this line
    time_info["end"] = time.time()
    detected = frame_info["end"] is not None
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

    # # video capture setup
    # cap = cv2.VideoCapture("fire.mp4")
    # vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # vid_bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))
    # detection_info = {"detected": False, "frame_count": -1, "time": -1}
    #
    # # process the first frame
    # # ret, first_frame = cap.read()
    # # prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # # mask = np.zeros_like(first_frame)
    #
    # # start 2 FPS timers
    # time_temp = time.time()
    # time_frame = {"start": time_temp, "prev": time_temp, "curr": time_temp}
    # fps_log = []
    # fps = FPS().start()
    #
    # for frame_id in tqdm(range(vid_len)):
    #     # read current frame
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     # frame time calculations
    #     fps.update()
    #     time_frame['curr'] = time.time()
    #     fps_live = 1 / (time_frame['curr'] - time_frame['prev'])
    #     time_frame['prev'] = time_frame['curr']
    #     fps_live = str(int(fps_live))
    #     frame = cv2.putText(frame, f"FPS: {fps_live}", (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (51, 87, 255), 1)
    #     fps_log.append(fps_live)
    #
    #     # process current frame
    #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     # TODO: call function to detect fire
    #     # detect.main(opt)
    #     # status, frame = detect_fire(gray)
    #     status = False # TODO: remove this line
    #     if status and detection_info["detected"] is False:
    #         detection_info["detected"] = True
    #         detection_info["frame_count"] = frame_id
    #         detection_info["time"] = time_frame['curr']
    #
    #     # show current frame
    #     # output = cv2.add(frame, mask)
    #     cv2.imshow("vid", frame)
    #     # canvas = np.zeros_like(first_frame)
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break
    # # clean up
    # fps.stop()
    # cap.release()
    # cv2.destroyAllWindows()
    #
    # print("Video information:")
    # print(f">Length: {vid_len} frames")
    # print(f">FPS: {vid_fps}")
    # print(f">Bitrate: {vid_bitrate}")
    # print("FPS information:")
    # print(f"[imutils] elasped time: {fps.elapsed():.2f} sec")
    # print(f"[imutils] approx. FPS: {fps.fps():.2f}")
    # print(f"[calculate] elasped time: {(time_frame['curr'] - time_frame['start']):.2f} sec")
    # print(f"[calculate] approx. FPS: {(vid_len / (time_frame['curr'] - time_frame['start'])):.2f}")
    # print("Detection information:")
    # print(f">Detected: {detection_info['detected']}")
    # print(f">Frame count: {detection_info['frame_count']}")
    # print(f">Time when detected: {detection_info['time']}")
    # return


if __name__ == '__main__':
    # opt = detect.parse_opt()
    main()
