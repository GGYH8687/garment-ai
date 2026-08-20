import os

import cv2
import numpy as np


# =========================
# 1. 路径与参数
# =========================
MASK_PATH = "inputs/garment_mask_hybrid.png"
FABRIC_PATH = "inputs/fabrics/fabric_01.jpg"
BASE_PATH = "outputs/base_shape.png"
OUTPUT_DIR = "outputs"

# 布料旋转（False=不旋转；True 时按 rotate_code 旋转）
rotate_fabric = False
rotate_code = cv2.ROTATE_90_CLOCKWISE

# 布料缩放（0.40=纹样较细密，已验证）
fabric_scale = 0.40

# 明暗保留强度（0=完全平贴；1=完全保留 base 明暗；0.5 中等）
shading_strength = 0.5


# =========================
# 2. 读取基础图、mask、布料
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


# 确保 mask 尺寸和 base 一致
mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


# 二值化 mask
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
mask_bool = mask > 0


# 布料旋转（所有 scale 共用）
if rotate_fabric:
    fabric = cv2.rotate(fabric, rotate_code)


# =========================
# 3. 预计算 base 的光影系数（所有 scale 共用，只算一次）
# =========================
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)

# 只在衣服区域统计平均亮度
garment_mean = base_gray[mask_bool].mean() if np.any(mask_bool) else 128.0

# 归一化成一个光影系数图
shading = base_gray / (garment_mean + 1e-6)

# 平滑一下，避免噪声太细
shading = cv2.GaussianBlur(shading, (21, 21), 0)

# 控制强弱，并裁剪范围
shading = 1.0 + (shading - 1.0) * shading_strength
shading = np.clip(shading, 0.65, 1.35)


# =========================
# 4. 平铺函数
# =========================
def tile_to_canvas(img, target_h, target_w):
    h, w = img.shape[:2]
    rep_y = target_h // h + 1
    rep_x = target_w // w + 1
    tiled = np.tile(img, (rep_y, rep_x, 1))
    return tiled[:target_h, :target_w]


# =========================
# 5. 用固定 fabric_scale 生成
# =========================
white_bg = np.full_like(base, 255)

print(f"\n--- fabric_scale={fabric_scale} ---")

# 处理 fabric：缩放
fh, fw = fabric.shape[:2]
new_w = max(8, int(fw * fabric_scale))
new_h = max(8, int(fh * fabric_scale))
fabric_scaled = cv2.resize(fabric, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# 平铺到画布
fabric_tiled = tile_to_canvas(fabric_scaled, H, W)

# 纯贴图版
texture_flat = base.copy()
texture_flat[mask_bool] = fabric_tiled[mask_bool]

# 保留明暗版
fabric_shaded = fabric_tiled.astype(np.float32) * shading[:, :, None]
fabric_shaded = np.clip(fabric_shaded, 0, 255).astype(np.uint8)
texture_shaded = base.copy()
texture_shaded[mask_bool] = fabric_shaded[mask_bool]

# 白底独立衣服版
texture_isolated = white_bg.copy()
texture_isolated[mask_bool] = fabric_shaded[mask_bool]

# 保存
cv2.imwrite(os.path.join(OUTPUT_DIR, "fabric_tiled.png"), fabric_tiled)
cv2.imwrite(os.path.join(OUTPUT_DIR, "texture_flat.png"), texture_flat)
cv2.imwrite(os.path.join(OUTPUT_DIR, "texture_shaded.png"), texture_shaded)
cv2.imwrite(os.path.join(OUTPUT_DIR, "texture_isolated.png"), texture_isolated)
print("Saved: fabric_tiled, texture_flat, texture_shaded, texture_isolated")

print("\nAll done!")
