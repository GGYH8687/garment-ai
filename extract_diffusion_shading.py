import cv2
import numpy as np


# =========================================================
# 路径
# =========================================================
# Stage 4.1 中 AI 干预最小的一张（strength=0.10）
AI_REFINED_PATH = "outputs/refined_strength_10.png"
# 原始精准 CV 贴图（保留这个的纹样）
WARPED_BASE_PATH = "outputs/texture_warped_best.png"
MASK_PATH = "inputs/garment_mask_clean.png"

SHADING_MAP_PATH = "outputs/diffusion_shading_map.png"
OUTPUT_PATH = "outputs/pattern_preserved_shading.png"


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
# 2. 提取 AI 图的低频光影（强模糊，去掉格纹等高频）
# =========================================================
BLUR_SIZE = (101, 101)

ai_gray = cv2.cvtColor(ai_refined, cv2.COLOR_BGR2GRAY).astype(np.float32)
ai_blurred = cv2.GaussianBlur(ai_gray, BLUR_SIZE, 0)


# =========================================================
# 3. 计算光影系数
#    以 AI 图服装区域的平均亮度为基准（=1.0）
#    亮处 > 1.0，暗处 < 1.0
# =========================================================
mean_light = ai_blurred[mask_bool].mean() if np.any(mask_bool) else 128.0
shading = ai_blurred / (mean_light + 1e-6)

# 限制范围，避免极端发黑/发白
shading = np.clip(shading, 0.85, 1.15)


# =========================================================
# 4. 保存光影地图（可视化用）
# =========================================================
shading_vis = ((shading - 0.85) / (1.15 - 0.85) * 255).astype(np.uint8)
cv2.imwrite(SHADING_MAP_PATH, shading_vis)


# =========================================================
# 5. 把光影乘到原始精准贴图上（保留 CV 纹样）
# =========================================================
warped_float = warped_base.astype(np.float32)
shaded_garment = warped_float * shading[:, :, None]
shaded_garment = np.clip(
    shaded_garment,
    0,
    255
)


# 只修改服装区域
result = warped_base.copy().astype(np.float32)
result[mask_bool] = shaded_garment[mask_bool]


# ---------------------------------------------------------
# 9. 保存最终结果
# ---------------------------------------------------------


result = result.astype(np.uint8)


cv2.imwrite(
    OUTPUT_PATH,
    result
)


print()
print("Stage 4.2 finished.")
print()
print("Shading map:")
print(SHADING_MAP_PATH)
print()
print("Pattern-preserved result:")
print(OUTPUT_PATH)
