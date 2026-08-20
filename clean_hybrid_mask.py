import cv2
import numpy as np


INPUT_PATH = "inputs/garment_mask_hybrid.png"
OUTPUT_PATH = "inputs/garment_mask_clean.png"


mask = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(INPUT_PATH)


_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)


# 1. 闭运算：补小洞
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)


# 2. 开运算：去小毛刺、小杂点
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)


# 3. 只保留最大连通区域
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)


if num_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)


cv2.imwrite(OUTPUT_PATH, mask)
print("Saved:", OUTPUT_PATH)
