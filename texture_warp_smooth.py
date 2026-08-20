import cv2
import numpy as np


# ========= 1. 路径 =========
MASK_PATH = "inputs/garment_mask_clean.png"
FABRIC_PATH = "inputs/fabrics/fabric_01.jpg"
BASE_PATH = "outputs/base_shape.png"
OUTPUT_PATH = "outputs/texture_warped_smooth.png"


# ========= 2. 读取 =========
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
top = ys.min()
bottom = ys.max()


fabric_h, fabric_w = fabric.shape[:2]


# ========= 3. 纹样重复次数 =========
repeat_x = 2.5
repeat_y = 2.5


# ========= 4. 收集每一行左右边界 =========
lefts = np.full(H, np.nan, dtype=np.float32)
rights = np.full(H, np.nan, dtype=np.float32)


for y in range(top, bottom + 1):
    row_x = np.where(mask[y] > 0)[0]
    if len(row_x) >= 2:
        lefts[y] = row_x.min()
        rights[y] = row_x.max()


# ========= 5. 对左右边界做平滑 =========
def smooth_nan_array(arr, window=15):
    arr2 = arr.copy()
    valid = ~np.isnan(arr2)


    # 先线性插值，把 nan 补上
    valid_idx = np.where(valid)[0]
    valid_val = arr2[valid]


    full_idx = np.arange(len(arr2))
    arr_interp = np.interp(full_idx, valid_idx, valid_val)


    # 再做滑动平均
    kernel = np.ones(window, dtype=np.float32) / window
    arr_smooth = np.convolve(arr_interp, kernel, mode="same")


    return arr_smooth


lefts_smooth = smooth_nan_array(lefts, window=15)
rights_smooth = smooth_nan_array(rights, window=15)


# ========= 6. 创建 remap 坐标 =========
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


        u = (x - left) / width
        u = np.clip(u, 0.0, 1.0)


        fx = (u * repeat_x * fabric_w) % fabric_w
        fy = (v * repeat_y * fabric_h) % fabric_h


        map_x[y, x] = fx
        map_y[y, x] = fy


# ========= 7. remap 纹理 =========
warped_fabric = cv2.remap(
    fabric,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_WRAP
)


# ========= 8. 加原衣服明暗 =========
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)


mean_light = base_gray[mask_bool].mean()
shading = base_gray / (mean_light + 1e-6)


shading = cv2.GaussianBlur(shading, (31, 31), 0)
shading = np.clip(shading, 0.75, 1.25)


warped_shaded = warped_fabric.astype(np.float32) * shading[:, :, None]
warped_shaded = np.clip(warped_shaded, 0, 255).astype(np.uint8)


# ========= 9. 合成 =========
result = base.copy()
result[mask_bool] = warped_shaded[mask_bool]


cv2.imwrite(OUTPUT_PATH, result)
print("Saved:", OUTPUT_PATH)
