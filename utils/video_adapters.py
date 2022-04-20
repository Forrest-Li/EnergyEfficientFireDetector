import cv2 as cv


class VideoWriterAdapter:
    def __init__(self, reader):
        self.write = cv.VideoWriter(reader.dir.replace(".mp4", "out.mp4"),
                                    cv.VideoWriter_fourcc(*'MP4V'),
                                    reader.fps(), reader.shape())

    def __call__(self, frame):
        self.write.write(frame)

    def release(self):
        self.write.release()


class VideoCaptureAdapter:
    def __init__(self, dir):
        self.dir = dir
        self.read = cv.VideoCapture(cv.samples.findFileOrKeep(dir))
        self._fps = self.read.get(cv.CAP_PROP_FPS)
        self._shape = (int(self.read.get(cv.CAP_PROP_FRAME_WIDTH)), \
                       int(self.read.get(cv.CAP_PROP_FRAME_HEIGHT)))

    def reset(self):
        self.read.release()
        self.read = cv.VideoCapture(cv.samples.findFileOrKeep(self.dir))

    def __call__(self):
        _, frame = self.read.read()
        return frame

    def fps(self):
        return self._fps

    def shape(self):
        return self._shape


class CachedVideoCaptureAdapter:
    def __init__(self, dir):
        self.dir = dir
        self.idx = 0
        read = VideoCaptureAdapter(dir)
        self._fps = read.fps()
        self._shape = read.shape()
        self.frames = []
        while True:
            f = read()
            if f is None:
                break
            self.frames.append(f)

    def reset(self):
        self.idx = 0

    def __call__(self):
        if self.idx < len(self.frames):
            f = self.frames[self.idx]
            self.idx += 1
            return f
        return None

    def fps(self):
        return self._fps

    def shape(self):
        return self._shape