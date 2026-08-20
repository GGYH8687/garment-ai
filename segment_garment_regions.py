import cv2
import numpy as np
import os


# =========================================================
# 路径
# =========================================================
MASK_PATH = "inputs/garment_mask_clean.png"
OUTPUT_DIR = "outputs/"


# =========================================================
# 1. 读取 garment_mask_clean.png
# =========================================================
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"找不到: {MASK_PATH}")

_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
mask_bool = mask > 0

H, W = mask.shape


# =========================================================
# 2. 找整体 bbox
# =========================================================
ys, xs = np.where(mask_bool)
y_top, y_bottom = int(ys.min()), int(ys.max())
x_left, x_right = int(xs.min()), int(xs.max())


# =========================================================
# 3. 用 y 坐标比例切上身 / 裙摆
#    连衣裙腰线大约在 50% 高度位置
# =========================================================
split_y = y_top + int((y_bottom - y_top) * 0.5)


# 上身区域
upper_part = np.zeros_like(mask)
upper_part[:split_y, :] = mask[:split_y, :]

# 裙摆区域
skirt = np.zeros_like(mask)
skirt[split_y:, :] = mask[split_y:, :]


# =========================================================
# 4. 在上身区域用列直方图找身体主体
#    （列像素数 >= max * 0.5 的连续区间 = 身体）
#    身体之外左侧 = 左袖，右侧 = 右袖
# =========================================================
col_sum = (upper_part > 0).sum(axis=0).astype(np.float32)
max_col = col_sum.max() if col_sum.max() > 0 else 1
threshold = max_col * 0.5

body_cols = col_sum >= threshold
body_xs = np.where(body_cols)[0]

if len(body_xs) > 0:
    body_x_left = int(body_xs.min())
    body_x_right = int(body_xs.max())
else:
    # fallback: 中间 60%
    body_x_left = x_left + int((x_right - x_left) * 0.2)
    body_x_right = x_right - int((x_right - x_left) * 0.2)


# =========================================================
# 5. 构造列掩码并广播到 H 行
# =========================================================
col_idx = np.arange(W)
left_col_mask = col_idx < body_x_left
body_col_mask = (col_idx >= body_x_left) & (col_idx <= body_x_right)
right_col_mask = col_idx > body_x_right

left_col_mask_full = np.broadcast_to(left_col_mask[None, :], (H, W))
body_col_mask_full = np.broadcast_to(body_col_mask[None, :], (H, W))
right_col_mask_full = np.broadcast_to(right_col_mask[None, :], (H, W))


# =========================================================
# 6. 切出 4 个区域
# =========================================================
upper_mask_bool = upper_part > 0

upper = np.where(upper_mask_bool & body_col_mask_full, 255, 0).astype(np.uint8)
sleeve_left = np.where(upper_mask_bool & left_col_mask_full, 255, 0).astype(np.uint8)
sleeve_right = np.where(upper_mask_bool & right_col_mask_full, 255, 0).astype(np.uint8)


# =========================================================
# 7. 保存
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)


cv2.imwrite(
    OUTPUT_DIR +
    "mask_upper.png",
    upper
)


cv2.imwrite(
    OUTPUT_DIR +
    "mask_left_sleeve.png",
    sleeve_left
)


cv2.imwrite(
    OUTPUT_DIR +
    "mask_right_sleeve.png",
    sleeve_right
)


cv2.imwrite(
    OUTPUT_DIR +
    "mask_skirt.png",
    skirt
)


print(
    "Stage 5.1 finished."
)
