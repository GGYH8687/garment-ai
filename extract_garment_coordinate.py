import cv2
import numpy as np
import csv


# =========================================================
# 路径
# =========================================================
MASK_PATH = "inputs/garment_mask_clean.png"

CENTER_OUTPUT = "outputs/garment_centerline.png"
WIDTH_OUTPUT = "outputs/garment_width_profile.png"
CSV_OUTPUT = "outputs/width_profile.csv"


# =========================================================
# 1. 读取 mask
# =========================================================
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"找不到: {MASK_PATH}")

_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
mask_bool = mask > 0

H, W = mask.shape


# =========================================================
# 5.3.1-A：提取左/右边界 + 中心线
# =========================================================
left_boundary = np.full(H, -1, dtype=np.int32)
right_boundary = np.full(H, -1, dtype=np.int32)

for y in range(H):
    row_x = np.where(mask[y] > 0)[0]
    if len(row_x) == 0:
        continue
    left_boundary[y] = int(row_x.min())
    right_boundary[y] = int(row_x.max())

# 中心线 = (left + right) / 2
center_profile = np.full(H, -1, dtype=np.float32)
for y in range(H):
    if left_boundary[y] >= 0 and right_boundary[y] >= 0:
        center_profile[y] = (left_boundary[y] + right_boundary[y]) / 2.0

# 平滑中心线（仅在有效行上）
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
for y in range(H):
    if left_boundary[y] >= 0:
        cx = int(round(center_smooth[y]))
        if 0 <= cx < W:
            # 画一条短的水平线段在中心位置
            length = min(20, width_profile[y] if width_profile[y] > 0 else 20)
            start = max(0, cx - length // 2)
            end = min(W, cx + length // 2)
            center_vis[y, start:end] = 255

# 直接画竖向中心线（连接每行中心点）
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
    # 用相对宽度画一条横向白线
    # 简单可视化：直接在 mask 区域内填白
    start = left_boundary[y]
    end = right_boundary[y]
    width_vis[y, start:end + 1] = 255

cv2.imwrite(WIDTH_OUTPUT, width_vis)


# =========================================================
# 6. 保存 CSV
# =========================================================
with open(
    CSV_OUTPUT,
    "w",
    newline=""
) as f:


    writer = csv.writer(f)


    writer.writerow(
        [
            "y",
            "left_boundary",
            "right_boundary",
            "width",
            "center"
        ]
    )


    for y in range(H):


        writer.writerow(
            [
                y,
                left_boundary[y],
                right_boundary[y],
                width_profile[y],
                center_profile[y]
            ]
            )


print(
    "Stage 5.3.1 finished."
)


print(
    CENTER_OUTPUT
)


print(
    WIDTH_OUTPUT
)


print(
    CSV_OUTPUT
)
