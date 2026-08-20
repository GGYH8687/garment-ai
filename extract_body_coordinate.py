import cv2
import numpy as np
import csv


# =========================================================
# 路径
# =========================================================
UPPER_PATH = "outputs/mask_upper_clean.png"
SKIRT_PATH = "outputs/mask_skirt_clean.png"

CENTER_OUTPUT = "outputs/body_centerline.png"
WIDTH_OUTPUT = "outputs/body_width_profile.png"
BOUNDARY_OUTPUT = "outputs/body_boundary_visual.png"
CSV_OUTPUT = "outputs/body_width_profile.csv"


# =========================================================
# 1. 读取 upper / skirt 并合并为 body mask
#    （不含袖子，避免腰部被袖子拉宽）
# =========================================================
upper = cv2.imread(UPPER_PATH, 0)
skirt = cv2.imread(SKIRT_PATH, 0)

if upper is None:
    raise FileNotFoundError(f"找不到: {UPPER_PATH}")
if skirt is None:
    raise FileNotFoundError(f"找不到: {SKIRT_PATH}")

H, W = upper.shape

body_mask = np.zeros_like(upper)
body_mask[upper > 127] = 255
body_mask[skirt > 127] = 255

_, body_mask = cv2.threshold(body_mask, 127, 255, cv2.THRESH_BINARY)
mask_bool = body_mask > 0


# =========================================================
# 5.3.1-A：提取左/右边界 + 中心线
# =========================================================
left_boundary = np.full(H, -1, dtype=np.int32)
right_boundary = np.full(H, -1, dtype=np.int32)

for y in range(H):
    row_x = np.where(body_mask[y] > 0)[0]
    if len(row_x) == 0:
        continue
    left_boundary[y] = int(row_x.min())
    right_boundary[y] = int(row_x.max())

# 中心线 = (left + right) / 2
center_profile = np.full(H, -1.0, dtype=np.float32)
for y in range(H):
    if left_boundary[y] >= 0 and right_boundary[y] >= 0:
        center_profile[y] = (left_boundary[y] + right_boundary[y]) / 2.0

# 平滑中心线
valid = center_profile >= 0
if np.any(valid):
    valid_idx = np.where(valid)[0]
    valid_val = center_profile[valid]
    full_idx = np.arange(H)
    center_smooth = np.interp(full_idx, valid_idx, valid_val)
else:
    center_smooth = np.full(H, W / 2.0, dtype=np.float32)


# =========================================================
# 5.3.1-B：计算每一行宽度
# =========================================================
width_profile = np.zeros(H, dtype=np.int32)
for y in range(H):
    if left_boundary[y] >= 0 and right_boundary[y] >= 0:
        width_profile[y] = right_boundary[y] - left_boundary[y] + 1


# =========================================================
# 5.3.1-C：生成可视化
# =========================================================

# --- 中心线可视化 ---
center_vis = np.zeros((H, W), dtype=np.uint8)
for y in range(1, H):
    if left_boundary[y] >= 0 and left_boundary[y - 1] >= 0:
        cx0 = int(round(center_smooth[y - 1]))
        cx1 = int(round(center_smooth[y]))
        if 0 <= cx0 < W and 0 <= cx1 < W:
            cv2.line(center_vis, (cx0, y - 1), (cx1, y), 255, 1)

cv2.imwrite(CENTER_OUTPUT, center_vis)


# --- 宽度 profile 可视化 ---
width_vis = np.zeros((H, W), dtype=np.uint8)
for y in range(H):
    if width_profile[y] <= 0:
        continue
    start = left_boundary[y]
    end = right_boundary[y]
    width_vis[y, start:end + 1] = 255

cv2.imwrite(WIDTH_OUTPUT, width_vis)


# --- 左右边界可视化（新增）---
boundary_img = np.zeros((H, W), dtype=np.uint8)

for y in range(H):
    if left_boundary[y] >= 0:
        boundary_img[y, left_boundary[y]] = 255
    if right_boundary[y] >= 0:
        boundary_img[y, right_boundary[y]] = 255

cv2.imwrite(BOUNDARY_OUTPUT, boundary_img)


# =========================================================
# 6. 保存 CSV
# =========================================================
with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["y", "left_boundary", "right_boundary", "width", "center"])
    for y in range(H):
        writer.writerow([
            y,
            left_boundary[y],
            right_boundary[y],
            width_profile[y],
            center_profile[y]
        ])


print("Stage 5.3.1 (body-only) finished.")
print(CENTER_OUTPUT)
print(WIDTH_OUTPUT)
print(BOUNDARY_OUTPUT)
print(CSV_OUTPUT)
