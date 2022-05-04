class BigLittleDetector:
  def __init__(self, little_detector, big_detector, rate):
    self.detectors = [little_detector, big_detector]
    self.rate = rate
  
  def detect(self, read) -> int:    
    active = 0
    while True:
      if not read.has_next():
        break
      if active == len(self.detectors):
        return read.frame_id()
      
      fire_frame = self.detectors[active].detect(read, self.rate[0])
      if fire_frame is not None:
        read.reset(fire_frame)
        active += 1
        continue
      elif active != 0:
        active = 0
        continue
      
      fire_frame = self.detectors[-1].detect(read, self.rate[1])
      if fire_frame is not None:
        read.reset(fire_frame)
        active += 1
        continue

    return None
      

#   def render(self, read):
#     self.read.reset()
#     fid = 0
#     while True:
#       print("frame:", fid)
#       fid += 1
#       img0 = self.read()
#       if img0 is None:
#           break
#       frame = self.frame_convert(img0)
#       pred = self.inference(frame)
      
#       self.plot_prediction(pred, img0, frame.shape)

#       self.write(img0)

#     self.write.release()