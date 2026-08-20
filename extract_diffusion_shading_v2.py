import cv2
import numpy as np


# =========================================================
# 路径
# =========================================================
# Stage 4.1 中 AI 干预最小的一张（strength=0.10）
AI_REFINED_PATH = "outputs/refined_strength_10.png"
# 原始精准 CV 贴图
WARPED_BASE_PATH = "outputs/texture_warped_best.png"
MASK_PATH = "inputs/garment_mask_clean.png"

SHADING_MAP_PATH = "outputs/diffusion_relative_shading_map_normalized.png"
OUTPUT_PATH = "outputs/pattern_preserved_relative_shading_normalized.png"


# =========================================================
# 1. 读取
# =========================================================
ai_refined = cv2.imread(AI_REFINED_PATH)
warped_base = cv2.imread(WARPED_BASE_PATH)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if ai_refined is None:
    raise FileNotFoundError(f"找不到: {AI_REFINED_PATH}")
if warped_base is None:
    raise FileNotFoundError(f"找不到: {WARPED_BASE_PATH}")
if mask is None:
    raise FileNotFoundError(f"找不到: {MASK_PATH}")

H, W = warped_base.shape[:2]
ai_refined = cv2.resize(ai_refined, (W, H), interpolation=cv2.INTER_LINEAR)
mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
mask_bool = mask > 0


# =========================================================
# 2. 提取两张图的低频亮度（强模糊，去高频纹样）
# =========================================================
BLUR_SIZE = (101, 101)

ai_gray = cv2.cvtColor(ai_refined, cv2.COLOR_BGR2GRAY).astype(np.float32)
warp_gray = cv2.cvtColor(warped_base, cv2.COLOR_BGR2GRAY).astype(np.float32)

ai_blurred = cv2.GaussianBlur(ai_gray, BLUR_SIZE, 0)
warp_blurred = cv2.GaussianBlur(warp_gray, BLUR_SIZE, 0)


# =========================================================
# 3. 相对光影 = AI低频亮度 ÷ Warp低频亮度
#    这表示：AI 相对于 Warp “额外”增加的明暗比例
#    = 1.0  → AI 没改变这里的光影
#    > 1.0  → AI 让这里额外变亮
#    < 1.0  → AI 让这里额外变暗
# =========================================================
relative_shading = ai_blurred / (warp_blurred + 1e-6)


# =========================================================
# 3.1 消除整体曝光变化
#
# 目的：
# AI 只能告诉我们
# "哪里应该相对亮一点 / 暗一点"
#
# 不允许 AI 把整件衣服一起变暗或变亮。
#
# 衣服区域的中位数重新设为 1.0
# =========================================================

median_shading = np.median(
    relative_shading[mask_bool]
)

print(
    "Median relative shading:",
    median_shading
)

relative_shading = (
    relative_shading /
    (median_shading + 1e-6)
)


# =========================================================
# 3.2 再限制变化范围
# =========================================================

relative_shading = np.clip(
    relative_shading,
    0.85,
    1.15
)


# =========================================================
# 4. 保存相对光影地图（可视化）
#    归一化到 0-255，中灰色(128) = AI没改变光影
# =========================================================
relative_vis = (
    (relative_shading - 0.85)
    / (1.15 - 0.85)
    * 255
).astype(np.uint8)

shading_vis = np.full_like(relative_vis, 128)  # 背景中灰
shading_vis[mask_bool] = (
    relative_vis[mask_bool]
)


cv2.imwrite(
    SHADING_MAP_PATH,
    shading_vis
)


# =========================================================
# 5. 把相对光影乘回精准 CV 纹样
# =========================================================
warped_float = (
    warped_base.astype(
        np.float32
    )
)


shaded = (
    warped_float
    *
    relative_shading[:, :, None]
)


shaded = np.clip(
    shaded,
    0,
    255
)


# ---------------------------------------------------------
# 11. 只修改衣服区域
# ---------------------------------------------------------


result = (
    warped_base.copy()
    .astype(np.float32)
)


result[mask_bool] = (
    shaded[mask_bool]
)


# ---------------------------------------------------------
# 12. 保存
# ---------------------------------------------------------


result = result.astype(
    np.uint8
)


cv2.imwrite(
    OUTPUT_PATH,
    result
)


print()
print(
    "Stage 4.3 finished."
)


print()


print(
    "Relative shading map:"
)


print(
    SHADING_MAP_PATH
)


print()


print(
    "Pattern-preserved result:"
)


print(
    OUTPUT_PATH
)
