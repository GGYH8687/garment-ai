import cv2
import numpy as np


OLD_MASK_PATH = "inputs/garment_mask.png"
ALIGNED_MASK_PATH = "inputs/garment_mask_aligned.png"
BASE_PATH = "outputs/base_shape.png"
OUTPUT_PATH = "inputs/garment_mask_hybrid.png"


old_mask = cv2.imread(OLD_MASK_PATH, cv2.IMREAD_GRAYSCALE)
aligned_mask = cv2.imread(ALIGNED_MASK_PATH, cv2.IMREAD_GRAYSCALE)
base = cv2.imread(BASE_PATH)


h, w = old_mask.shape
aligned_mask = cv2.resize(aligned_mask, (w, h), interpolation=cv2.INTER_NEAREST)


_, old_mask = cv2.threshold(old_mask, 127, 255, cv2.THRESH_BINARY)
_, aligned_mask = cv2.threshold(aligned_mask, 127, 255, cv2.THRESH_BINARY)


# 分界线：大约衣服上部 35%
split_y = int(h * 0.35)


hybrid = np.zeros_like(old_mask)


# 上部用旧 mask
hybrid[:split_y, :] = old_mask[:split_y, :]


# 下部用 aligned mask
hybrid[split_y:, :] = aligned_mask[split_y:, :]


# --------------------------------
# 清理：去小杂点 + 填小洞
# --------------------------------
# 开运算：去掉小的白色杂点
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
hybrid = cv2.morphologyEx(hybrid, cv2.MORPH_OPEN, kernel_open)

# 闭运算：填小洞、补缝隙（用更大的核）
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
hybrid = cv2.morphologyEx(hybrid, cv2.MORPH_CLOSE, kernel_close)


# --------------------------------
# 补齐腰部两侧：
#   在腰部区域（split_y ~ 65% 高度）扩大搜索范围，
#   把 base_shape 中的白色衣服区域补进来
# --------------------------------
waist_top = int(h * 0.35)
waist_bot = int(h * 0.65)
waist_band = np.zeros_like(hybrid)
waist_band[waist_top:waist_bot, :] = 255

# 腰部附近扩大搜索区域
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
search_area = cv2.dilate(hybrid, kernel_dilate, iterations=1)
search_area = cv2.bitwise_and(search_area, waist_band)

# base_shape 中的白色衣服区域
hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 150])
upper_white = np.array([180, 100, 255])
white_region = cv2.inRange(hsv, lower_white, upper_white)

# 腰部白色衣服 = 搜索区 ∩ 白色区域
extra_waist = cv2.bitwise_and(white_region, search_area)

# 合并到 hybrid
hybrid = cv2.bitwise_or(hybrid, extra_waist)

# 再次闭运算平滑
hybrid = cv2.morphologyEx(hybrid, cv2.MORPH_CLOSE, kernel_close)


# --------------------------------
# 只保留最大连通区域，去零散块
# --------------------------------
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hybrid, connectivity=8)
if num_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    hybrid = np.where(labels == largest_label, 255, 0).astype(np.uint8)


cv2.imwrite(OUTPUT_PATH, hybrid)
print("Saved:", OUTPUT_PATH)
