import cv2
import numpy as np


MASK_PATH = "outputs/garment_mask.png"
OUTLINE_PATH = "outputs/garment_outline.png"
CLEAN_EDGE_PATH = "outputs/garment_edge_clean.png"
HYBRID_PATH = "outputs/garment_hybrid_control.png"

mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
outline = cv2.imread(OUTLINE_PATH, cv2.IMREAD_GRAYSCALE)
clean_edge = cv2.imread(CLEAN_EDGE_PATH, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(f"无法读取 {MASK_PATH}")
if outline is None:
    raise FileNotFoundError(f"无法读取 {OUTLINE_PATH}")
if clean_edge is None:
    raise FileNotFoundError(f"无法读取 {CLEAN_EDGE_PATH}")


# =========================
# 找到衣服主体的 bounding box
# =========================
ys, xs = np.where(mask > 0)
if len(xs) == 0 or len(ys) == 0:
    raise ValueError("mask 中没有检测到白色区域")

x1, x2 = xs.min(), xs.max()
y1, y2 = ys.min(), ys.max()

h = y2 - y1
w = x2 - x1

# 稍微向上和向左右扩一点，方便保留领口/肩部
pad_x = int(0.10 * w)
pad_top = int(0.18 * h)

x1e = max(0, x1 - pad_x)
x2e = min(mask.shape[1] - 1, x2 + pad_x)
y1e = max(0, y1 - pad_top)


# =========================
# 上半部分区域：从 clean_edge 里取
# 下半部分区域：从 outline 里取
# =========================
hybrid = np.zeros_like(mask)

# 上部 40%
split_y = y1 + int(0.40 * h)

# 1) 先放入衣服 outline，作为整体外轮廓
hybrid = cv2.bitwise_or(hybrid, outline)

# 2) 从 clean edge 中拿回“上半部分细节”
top_region = np.zeros_like(mask)
top_region[y1e:split_y, x1e:x2e + 1] = 255

top_edges = cv2.bitwise_and(clean_edge, top_region)

# 为了减少上半部分残留手臂噪声：
# 只保留靠近图像中心的边
center_x = (x1 + x2) // 2
max_half_width = int(0.38 * w)

filtered_top = np.zeros_like(mask)
for y in range(y1e, split_y):
    xs_row = np.where(top_edges[y] > 0)[0]
    if len(xs_row) == 0:
        continue

    xs_keep = xs_row[np.abs(xs_row - center_x) <= max_half_width]
    filtered_top[y, xs_keep] = 255

# 3) 合并
hybrid = cv2.bitwise_or(hybrid, filtered_top)


# =========================
# 去除可能的孤立小噪点
# =========================
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    hybrid, connectivity=8
)

cleaned = np.zeros_like(hybrid)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= 8:
        cleaned[labels == i] = 255

hybrid = cleaned


# =========================
# 稍微加粗一点线条，便于 ControlNet 看清
# =========================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
hybrid = cv2.dilate(hybrid, kernel, iterations=1)


# =========================
# 保存
# =========================
cv2.imwrite(HYBRID_PATH, hybrid)

print("已生成：", HYBRID_PATH)
print(f"mask bbox: ({x1}, {y1}) - ({x2}, {y2})")
print(f"hybrid top region: y={y1e}..{split_y}, x={x1e}..{x2e}")
