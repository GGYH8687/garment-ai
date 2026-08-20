import cv2
import numpy as np


# =========================
# 1. 路径与参数
# =========================
MASK_PATH = "inputs/garment_mask_clean.png"
FABRIC_PATH = "inputs/fabrics/fabric_01.jpg"
BASE_PATH = "outputs/base_shape.png"
OUTPUT_DIR = "outputs"

# 纹样重复次数
repeat_x = 2.5
repeat_y = 2.5

# 轻度变形强度（0=完全平整；1=完全按衣服宽度压缩）
warp_strengths = [0.15, 0.25]

# 边界平滑窗口
window = 15


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

ys, xs = np.where(mask_bool)
if len(ys) == 0:
    raise RuntimeError("mask 全黑")
top, bottom = ys.min(), ys.max()

fabric_h, fabric_w = fabric.shape[:2]


# =========================
# 3. 收集每一行左右边界
# =========================
lefts = np.full(H, np.nan, dtype=np.float32)
rights = np.full(H, np.nan, dtype=np.float32)

for y in range(top, bottom + 1):
    row_x = np.where(mask[y] > 0)[0]
    if len(row_x) >= 2:
        lefts[y] = row_x.min()
        rights[y] = row_x.max()


# =========================
# 4. 平滑边界（线性插值 + 滑动平均）
# =========================
def smooth_nan_array(arr, win=15):
    arr2 = arr.copy()
    valid = ~np.isnan(arr2)
    valid_idx = np.where(valid)[0]
    valid_val = arr2[valid]
    full_idx = np.arange(len(arr2))
    arr_interp = np.interp(full_idx, valid_idx, valid_val)
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(arr_interp, kernel, mode="same")


lefts_smooth = smooth_nan_array(lefts, window)
rights_smooth = smooth_nan_array(rights, window)


# =========================
# 4.5 预计算 shading（所有 warp_strength 共用）
# =========================
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
mean_light = base_gray[mask_bool].mean()
shading = base_gray / (mean_light + 1e-6)
shading = cv2.GaussianBlur(shading, (31, 31), 0)
shading = np.clip(shading, 0.78, 1.22)


# 整张画布的平整坐标（不随衣服宽度变化，循环外算一次）
flat_u_norm = np.tile(np.arange(W, dtype=np.float32), (H, 1)) / W


# =========================
# 5-8. 遍历 warp_strength 生成对照
# =========================
for warp_strength in warp_strengths:
    tag = f"ws{int(warp_strength * 100):02d}"
    print(f"\n--- warp_strength={warp_strength} ({tag}) ---")

    # ---- 5. Gentle Warp remap 坐标 ----
    map_x = np.zeros((H, W), dtype=np.float32)
    map_y = np.zeros((H, W), dtype=np.float32)

    for y in range(top, bottom + 1):
        left = lefts_smooth[y]
        right = rights_smooth[y]
        width = right - left
        if width <= 1:
            continue

        v = (y - top) / max(bottom - top, 1)

        row_x = np.where(mask[y] > 0)[0]
        if len(row_x) == 0:
            continue

        real_left = row_x.min()
        real_right = row_x.max()

        for x in range(real_left, real_right + 1):
            if mask[y, x] == 0:
                continue

            u_flat = flat_u_norm[y, x]
            u_warp = (x - left) / width
            u = (1.0 - warp_strength) * u_flat + warp_strength * u_warp
            u = np.clip(u, 0.0, 1.0)

            fx = (u * repeat_x * fabric_w) % fabric_w
            fy = (v * repeat_y * fabric_h) % fabric_h

            map_x[y, x] = fx
            map_y[y, x] = fy

    # ---- 6. remap 纹理 ----
    warped = cv2.remap(
        fabric,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )

    # ---- 7. 加原衣服明暗 ----
    warped_shaded = warped.astype(np.float32) * shading[:, :, None]
    warped_shaded = np.clip(warped_shaded, 0, 255).astype(np.uint8)

    # ---- 8. 合成 ----
    result = base.copy()
    result[mask_bool] = warped_shaded[mask_bool]

    out_path = f"{OUTPUT_DIR}/texture_warped_gentle_{tag}.png"
    cv2.imwrite(out_path, result)
    print("Saved:", out_path)

print("\nAll done!")
