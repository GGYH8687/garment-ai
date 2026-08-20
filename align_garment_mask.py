import cv2
import numpy as np


BASE_PATH = "outputs/base_shape.png"
OLD_MASK_PATH = "inputs/garment_mask.png"
OUTPUT_PATH = "inputs/garment_mask_aligned.png"


# 读取图片
base = cv2.imread(BASE_PATH)
mask = cv2.imread(OLD_MASK_PATH, cv2.IMREAD_GRAYSCALE)


# 保证尺寸一致
h, w = base.shape[:2]
mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)


# --------------------------------
# 1. 扩大原 mask，建立“搜索区域”
# --------------------------------
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (31, 31)
)


search_area = cv2.dilate(
    mask,
    kernel,
    iterations=1
)


# --------------------------------
# 2. 找 base_shape 中偏白的区域
# --------------------------------
hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)


# 白色通常：
# 饱和度低 S
# 亮度高 V
lower_white = np.array([0, 0, 150])
upper_white = np.array([180, 100, 255])


white_region = cv2.inRange(
    hsv,
    lower_white,
    upper_white
)


# --------------------------------
# 3. 只在原衣服附近寻找白色区域
# 避免把背景也选进来
# --------------------------------
extra_garment = cv2.bitwise_and(
    white_region,
    search_area
)


# 原 mask + 新检测区域
aligned_mask = cv2.bitwise_or(
    mask,
    extra_garment
)


# --------------------------------
# 4. 填小洞
# --------------------------------
kernel_close = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (9, 9)
)


aligned_mask = cv2.morphologyEx(
    aligned_mask,
    cv2.MORPH_CLOSE,
    kernel_close
)


cv2.imwrite(OUTPUT_PATH, aligned_mask)


print("Saved:", OUTPUT_PATH)
