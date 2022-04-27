class FirePolicy:
    def __init__(self, fps, time_window, thres_percentage, step=-1):
        lenOfwin = min(step, int(fps * time_window)
                       ) if step > 0 else int(fps * time_window)
        self.frame_window = [0 for i in range(lenOfwin)]
        assert thres_percentage >= 0. and thres_percentage <= 1.
        self.thres = max(1, int(thres_percentage * len(self.frame_window)))
        # print(f"source fps: {fps}, threshold: {self.thres} / {len(self.frame_window)}")
        self.num_fire_frames = 0

    def make_policy(time_window, thres_percentage):
        return lambda fps, step=-1: FirePolicy(fps, time_window, thres_percentage)

    def update_frame_record(self, fid, has_fire) -> int:
        fid_ = fid % len(self.frame_window)
        self.num_fire_frames -= self.frame_window[fid_]
        self.frame_window[fid_] = int(has_fire)
        self.num_fire_frames += self.frame_window[fid_]

        if self.num_fire_frames >= self.thres:
            l = len(self.frame_window)
            for i in range(max(0, fid - l + 1), fid):
                if (self.frame_window[i % l] == 1):
                    return i

        return -1

    def frame_len(self) -> int:
        return len(self.frame_window)
