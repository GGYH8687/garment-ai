import cv2
import numpy as np


# =========================================================
# 路径
# =========================================================
BASE_SHAPE_PATH = "outputs/base_shape.png"
WARPED_PATH = "outputs/texture_warped_best.png"
MASK_PATH = "inputs/garment_mask_clean.png"

DETAIL_MAP_PATH = (
    "outputs/base_detail_map_normalized_maskaware.png"
)


OUTPUT_PATH = (
    "outputs/pattern_preserved_base_detail_normalized_maskaware.png"
)


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
# 2. 真正的 Mask-Aware Gaussian Blur (Normalized Convolution)
#
#   Blur(image * mask)
#   ------------------
#        Blur(mask)
#
# 背景权重为 0，
# 所以背景完全不参与衣服光影估计。
# =========================================================
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)

BLUR_SIZE = (51, 51)


# mask:
#   0   → 背景
#   255 → 衣服
# 转为：
#   0.0 → 背景
#   1.0 → 衣服
mask_float = (
    mask.astype(np.float32)
    / 255.0
)


# ---------------------------------------------------------
# 1. 只有衣服像素参与
# ---------------------------------------------------------
weighted_image = (
    base_gray *
    mask_float
)


# ---------------------------------------------------------
# 2. 对衣服亮度求 Gaussian Blur
# ---------------------------------------------------------
blurred_image = cv2.GaussianBlur(
    weighted_image,
    BLUR_SIZE,
    0
)


# ---------------------------------------------------------
# 3. 同时 Blur mask
#
# 它相当于告诉我们：
# 每个位置到底有多少有效衣服像素参与。
# ---------------------------------------------------------
blurred_mask = cv2.GaussianBlur(
    mask_float,
    BLUR_SIZE,
    0
)


# ---------------------------------------------------------
# 4. Normalized Convolution
#
# 这是本阶段最关键的一行。
# ---------------------------------------------------------
base_blurred = (
    blurred_image /
    (blurred_mask + 1e-6)
)


# 细节比例
detail_ratio = base_gray / (base_blurred + 1e-6)

# 归一化中位数到 1.0
median_detail = np.median(detail_ratio[mask_bool])
detail_ratio = detail_ratio / (median_detail + 1e-6)

# 限制范围（细节是轻量明暗变化）
detail_ratio = np.clip(detail_ratio, 0.95, 1.05)


# ---------------------------------------------------------


detail_vis = (
    (detail_ratio - 0.95)
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
# 11. 乘回精准纹样
# ---------------------------------------------------------


warped_float = (
    warped.astype(np.float32)
)


detailed = (
    warped_float
    *
    detail_ratio[:, :, None]
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
# 12. 保存
# ---------------------------------------------------------


cv2.imwrite(
    OUTPUT_PATH,
    result
)


print()


print("Stage 4.5 v3 (normalized mask-aware) finished.")


print()


print("Mask-aware detail map:")
print(DETAIL_MAP_PATH)


print()


print("Pattern-preserved result:")
print(OUTPUT_PATH)
