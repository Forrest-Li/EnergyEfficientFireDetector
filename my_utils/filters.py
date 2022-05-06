import numpy as np
import cv2 as cv
from numba import jit


def un_morph_filter(filter):
    def new_filter(img, mask):
        mask = filter(img, mask)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        return mask

    return new_filter


def OF_fire_filter(two_frames, mask):
    img1, img2 = two_frames[0], two_frames[1]
    img1_r = img1[:, :, 2]
    img2_r = img2[:, :, 2]


@un_morph_filter
def fire_filter(img, mask):
    r_mean = np.mean(img[:, :, 2], dtype=np.float32)
    r_chn = img[:, :, 2]
    g_chn = img[:, :, 1]
    b_chn = img[:, :, 0]
    r_1_chn = r_chn.copy()
    r_1_chn[r_1_chn == 0] = 1
    g_1_chn = g_chn.copy()
    g_1_chn[g_1_chn == 0] = 1

    g_r_1 = g_chn / r_1_chn
    b_r_1 = b_chn / r_1_chn
    b_g_1 = b_chn / g_1_chn
    where = np.where(((mask == 255)
                      & (r_chn > r_mean)
                      & (r_chn > g_chn) & (g_chn > b_chn)
                      & (g_r_1 <= 0.65)
                      #& (g_r_1 >= 0.25)
                      #& (b_r_1 >= 0.05) & (b_r_1 <= 0.45) #&
                      #(b_g_1 >= 0.2) & (b_g_1 <= 0.6)
                      ))

    new_mask = np.zeros_like(mask)
    new_mask[where] = 255
    return new_mask


# @jit(nopython=True)
# def filter_jit(img, mask):
#   r_mean = np.float32(np.mean(img[:, :, 2]))

#   M, N = mask.shape
#   for i in range(M):
#     for j in range(N):
#       if mask[i, j] != 255:
#         continue
#       if not (img[i, j, 2] > r_mean and \
#       img[i, j, 2] > img[i, j, 1] and img[i, j, 1] > img[i, j, 0]):
#         mask[i, j] = 0
#         continue
#       r1 = np.float32(img[i, j, 2]) + 1
#       e1 = np.float32(img[i, j, 1]) / r1
#       if not (e1 < np.float32(0.65) and e1 > np.float32(0.25)):
#         mask[i, j] = 0
#         continue
#       e2 = np.float32(img[i, j, 0]) / r1
#       if not (e2 > np.float32(0.05) and e2 < np.float32(0.45)):
#         mask[i, j] = 0
#         continue

#       e3 = np.float32(img[i, j, 0]) / (np.float32(img[i, j, 1]) + 1)
#       if not (e3 > np.float32(0.2) and e3 < np.float32(0.6)):
#         mask[i, j] = 0
#         continue


@un_morph_filter
@jit(nopython=True)
def filter_jit(img, mask):
    r_mean = np.float32(np.mean(img[:, :, 2]))

    M, N = mask.shape
    for i in range(M):
        for j in range(N):
            if mask[i, j] != 255:
                continue
            r1 = np.float32(img[i, j, 2]) + 1
            e1 = np.float32(img[i, j, 1]) / r1
            e2 = np.float32(img[i, j, 0]) / r1
            e3 = np.float32(img[i, j, 0]) / (np.float32(img[i, j, 1]) + 1)
            if not (img[i, j, 2] > r_mean and
                    img[i, j, 2] > img[i, j, 1] and img[i, j, 1] > img[i, j, 0]
                    and e1 < 0.65 and e1 > 0.25 and e2 > 0.05 and e2 < 0.45
                    and e3 > 0.2 and e3 < 0.6):
                mask[i, j] = 0
                continue
    return mask


@un_morph_filter
def filter_non_jit(img, mask):
    r_mean = np.float32(np.mean(img[:, :, 2]))

    M, N = mask.shape
    for i in range(M):
        for j in range(N):
            if mask[i, j] != 255:
                continue
            r1 = np.float32(img[i, j, 2]) + 1
            e1 = np.float32(img[i, j, 1]) / r1
            e2 = np.float32(img[i, j, 0]) / r1
            e3 = np.float32(img[i, j, 0]) / (np.float32(img[i, j, 1]) + 1)
            if not (img[i, j, 2] > r_mean and
                    img[i, j, 2] > img[i, j, 1] and img[i, j, 1] > img[i, j, 0]
                    and e1 < 0.65 and e1 > 0.25 and e2 > 0.05 and e2 < 0.45
                    and e3 > 0.2 and e3 < 0.6):
                mask[i, j] = 0
                continue
    return mask
