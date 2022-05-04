import numpy as np
import cv2 as cv
from scipy.spatial import ConvexHull


# def benchmarking_jits(jit_func):
#   def renderer(img, mask):
#     jit_func(img, mask)
#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
#     where = np.where(mask == 255)
#     return where, mask
#   return renderer


class FireHandle:
  def __init__(self, mask_gen, thres):
    self.mask_gen = mask_gen
    self.thres = thres

  def has_fire(self, img, mask):
    mask = self.mask_gen(img, mask)
    # size = int(mask.shape[0] * mask.shape[1] * self.thres)
    return mask[mask == 255].shape[0] > 0

  def calc_area(self, corners):
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
      j = (i + 1) % n
      area += corners[i][0] * corners[j][1]
      area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

  def apply_mask(self, img, mask):
    mask = self.mask_gen(img, mask)
    where = np.where(mask == 255)
    points = np.array([where[1], where[0]]).transpose()
    # rect = cv.minAreaRect(points)
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    if len(points) <= 1:
      return img, 0.
    hull = ConvexHull(points)
    hull_points = [points[id] for id in hull.vertices]
    img = cv.polylines(img, np.array([hull_points]), True, (125, 206, 160), 5)
    area = self.calc_area(hull_points)
    ratio = area / (img.shape[0] * img.shape[1])

    # return cv.drawContours(img, [box], 0, (0, 0, 255), 2)
    return img, ratio