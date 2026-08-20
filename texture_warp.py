import cv2
import numpy as np


# =========================
# 1. 路径与参数
# =========================
MASK_PATH = "inputs/garment_mask_hybrid.png"
FABRIC_PATH = "inputs/fabrics/fabric_01.jpg"
BASE_PATH = "outputs/base_shape.png"
OUTPUT_PATH = "outputs/texture_warped.png"

# 纹样重复次数（越大格纹越小/越密）
repeat_x = 2.5
repeat_y = 2.5

# 明暗保留强度
shading_strength = 0.5


# =========================
# 2. 读取
# =========================
base = cv2.imread(BASE_PATH)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
fabric = cv2.imread(FABRIC_PATH)

if base is None:
    raise FileNotFoundError(f"找不到: {BASE_PATH}")
if mask is None:
    raise FileNotFoundError(f"找不到: {MASK_PATH}")
if fabric is None:
    raise FileNotFoundError(f"找不到: {FABRIC_PATH}")

H, W = base.shape[:2]
mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
mask_bool = mask > 0


# =========================
# 3. 准备一块布料（按 repeat 缩放）
#    先把 fabric 缩放到 (tile_w, tile_h)，再后面逐行重映射
# =========================
fh, fw = fabric.shape[:2]
tile_w = max(8, int(fw / repeat_x))
tile_h = max(8, int(fh / repeat_y))
fabric_tile = cv2.resize(fabric, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)


# =========================
# 4. 预计算 base 光影系数（所有行共用）
# =========================
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
garment_mean = base_gray[mask_bool].mean() if np.any(mask_bool) else 128.0
shading = base_gray / (garment_mean + 1e-6)
shading = cv2.GaussianBlur(shading, (21, 21), 0)
shading = 1.0 + (shading - 1.0) * shading_strength
shading = np.clip(shading, 0.65, 1.35)


# =========================
# 5. 逐行 Texture Warping
#    对每一横行：找衣服左/右边界，把 fabric_tile 整行重映射到该宽度
#    竖直方向也按衣服高度重复 repeat_y 次
# =========================
# 找到衣服的上下边界
ys, xs = np.where(mask_bool)
if len(ys) == 0:
    raise RuntimeError("mask 全黑，没有衣服区域")
top_y, bot_y = ys.min(), ys.max()
garment_h = bot_y - top_y + 1

# --------------------------------
# 5.1 先收集每一行的 left/right
# --------------------------------
lefts = np.full(H, -1, dtype=np.int32)
rights = np.full(H, -1, dtype=np.int32)
for y in range(top_y, bot_y + 1):
    row_x = np.where(mask_bool[y])[0]
    if len(row_x) == 0:
        continue
    lefts[y] = row_x.min()
    rights[y] = row_x.max()

# --------------------------------
# 5.2 滑动平均平滑左右边界（window=15）
# --------------------------------
window = 15
half = window // 2


def smooth_boundaries(arr):
    smoothed = arr.copy()
    for y in range(top_y, bot_y + 1):
        y0 = max(top_y, y - half)
        y1 = min(bot_y, y + half)
        # 只取有效行（!=-1）
        valid = arr[y0:y1 + 1]
        valid = valid[valid != -1]
        if len(valid) > 0:
            smoothed[y] = int(np.mean(valid))
    return smoothed


lefts = smooth_boundaries(lefts)
rights = smooth_boundaries(rights)

# --------------------------------
# 5.3 逐行 warp
# --------------------------------
warped = np.zeros_like(base)

for y in range(top_y, bot_y + 1):
    left = lefts[y]
    right = rights[y]
    if left < 0 or right < 0:
        continue
    row_w = right - left + 1
    if row_w < 2:
        continue

    # 竖直方向：当前行对应 fabric_tile 的哪个 y
    fabric_y = int(((y - top_y) / garment_h) * tile_h) % tile_h
    fabric_row = fabric_tile[fabric_y:fabric_y + 1, :, :]  # (1, tile_w, 3)

    # 横向：把 fabric_row 整行重映射到 row_w 宽度（这就是 warp 核心）
    warped_row = cv2.resize(fabric_row, (row_w, 1), interpolation=cv2.INTER_LINEAR)

    # 写入该行（仅左右边界之间）
    warped[y, left:right + 1, :] = warped_row[0]


# =========================
# 6. 应用光影
# =========================
warped_shaded = np.clip(
    warped.astype(np.float32) * shading[:, :, None],
    0,
    255
).astype(np.uint8)


# =========================
# 7. 合成
# =========================
result = base.copy()
result[mask_bool] = warped_shaded[mask_bool]


# =========================
# 8. 保存
# =========================
cv2.imwrite(
    OUTPUT_PATH,
    result
)

print("Saved:", OUTPUT_PATH)
