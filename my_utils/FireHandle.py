import numpy as np
import cv2 as cv


class FireHandle:
    def __init__(self, mask_gen, thres):
        self.mask_gen = mask_gen
        self.thres = thres

    def has_fire(self, img, mask):
        mask = self.mask_gen(img, mask)
        # size = int(mask.shape[0] * mask.shape[1] * self.thres)
        return mask[mask == 255].shape[0] > 1

    def apply_mask(self, img, mask):
        mask = self.mask_gen(img, mask)
        where = np.where(mask == 255)
        rect = cv.minAreaRect(
            np.array([where[1], where[0]]).transpose().reshape(-1, 1, 2))
        box = cv.boxPoints(rect)
        box = np.int0(box)
        return cv.drawContours(img, [box], 0, (0, 0, 255), 2)
