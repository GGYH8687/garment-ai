import cv2
import numpy as np


# =========================================================
# 路径
# =========================================================
BASE_SHAPE_PATH = "outputs/base_shape.png"
WARPED_PATH = "outputs/texture_warped_best.png"
MASK_PATH = "inputs/garment_mask_clean.png"

DETAIL_MAP_PATH = "outputs/base_detail_map.png"
OUTPUT_PATH = "outputs/pattern_preserved_base_detail.png"


# =========================================================
# 1. 读取
# =========================================================
base = cv2.imread(BASE_SHAPE_PATH)
warped = cv2.imread(WARPED_PATH)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if base is None:
    raise FileNotFoundError(f"找不到: {BASE_SHAPE_PATH}")
if warped is None:
    raise FileNotFoundError(f"找不到: {WARPED_PATH}")
if mask is None:
    raise FileNotFoundError(f"找不到: {MASK_PATH}")

H, W = warped.shape[:2]
base = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)
mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
mask_bool = mask > 0


# =========================================================
# 2. 从 base_shape 提取高频细节层
#
# 思路：
#   base_shape = 低频(大范围光影) + 高频(细节)
#   细节层 = base_gray / base_blurred
#   = 1.0  → 该处无细节
#   > 1.0  → 高光
#   < 1.0  → 阴影/折线
# =========================================================
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)

# 较小的模糊核，保留"中等尺度"的细节（腰线、衣片、接缝）
BLUR_SIZE = (51, 51)
base_blurred = cv2.GaussianBlur(base_gray, BLUR_SIZE, 0)

# 归一化：以衣服区域中位数为基准
median_base = np.median(base_blurred[mask_bool])
base_blurred_norm = base_blurred / (median_base + 1e-6)

detail_layer = base_gray / (base_blurred + 1e-6)

# 归一化 detail_layer 的中位数到 1.0
median_detail = np.median(detail_layer[mask_bool])
detail_layer = detail_layer / (median_detail + 1e-6)

# 限制范围：细节是轻量的明暗变化，不允许极端值
detail_layer = np.clip(detail_layer, 0.95, 1.05)


# ---------------------------------------------------------
# 9. 保存 Detail Map
#
# 中灰 = 没变化
# 亮 = 高光
# 暗 = 阴影
# ---------------------------------------------------------


detail_vis = (
    (detail_layer - 0.95)
    /
    (1.05 - 0.95)
    * 255
)


detail_vis = np.clip(
    detail_vis,
    0,
    255
).astype(np.uint8)


detail_map = np.full(
    (H, W),
    128,
    dtype=np.uint8
)


detail_map[mask_bool] = (
    detail_vis[mask_bool]
)


cv2.imwrite(
    DETAIL_MAP_PATH,
    detail_map
)


# ---------------------------------------------------------
# 10. 乘回精准纹样
# ---------------------------------------------------------


warped_float = warped.astype(
    np.float32
)


detailed = (
    warped_float
    *
    detail_layer[:, :, None]
)


detailed = np.clip(
    detailed,
    0,
    255
)


result = warped_float.copy()


result[mask_bool] = (
    detailed[mask_bool]
)


result = result.astype(
    np.uint8
)


# ---------------------------------------------------------
# 11. 保存
# ---------------------------------------------------------


cv2.imwrite(
    OUTPUT_PATH,
    result
)


print()
print("Stage 4.4 finished.")


print()
print("Detail map:")
print(DETAIL_MAP_PATH)


print()
print("Result:")
print(OUTPUT_PATH)
