import cv2
import numpy as np


INPUT_PATH = "outputs/garment_edge_clean.png"
MASK_PATH = "outputs/garment_mask.png"
OUTLINE_PATH = "outputs/garment_outline.png"
CONTROL_PATH = "outputs/garment_control.png"

MIN_WIDTH = 60
MAX_JUMP = 35


# =========================
# 1. 读取干净 Canny 边缘图
# =========================
edge = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)

if edge is None:
    raise FileNotFoundError(f"无法读取 {INPUT_PATH}")

_, edge = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)


# =========================
# 2. 逐行寻找最靠近中心线的左右边界
#    尽量避开更靠外侧的手臂轮廓
# =========================
height, width = edge.shape
center_x = width // 2
mask = np.zeros_like(edge)

prev_left = None
prev_right = None

for y in range(height):
    x_positions = np.where(edge[y] > 0)[0]
    left_candidates = x_positions[x_positions < center_x]
    right_candidates = x_positions[x_positions > center_x]

    if len(left_candidates) == 0 or len(right_candidates) == 0:
        continue

    left = int(left_candidates[-1])
    right = int(right_candidates[0])

    # =========================
    # 3. 宽度过滤与连续性约束
    # =========================
    if right - left < MIN_WIDTH:
        continue

    if prev_left is not None and abs(left - prev_left) > MAX_JUMP:
        left = prev_left
    if prev_right is not None and abs(right - prev_right) > MAX_JUMP:
        right = prev_right

    if right - left < MIN_WIDTH:
        continue

    mask[y, left:right + 1] = 255
    prev_left = left
    prev_right = right


# =========================
# 4. 形态学处理
#    close：填小洞、补缝隙
#    open ：去小毛刺
# =========================
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)


# =========================
# 5. 只保留最大的连通区域
#    防止零散噪声块残留
# =========================
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    mask, connectivity=8
)

if num_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)


# =========================
# 6. 去掉裙摆以下可能残留的腿部
# =========================
row_widths = np.sum(mask > 0, axis=1)
valid_rows = np.where(row_widths > 0)[0]

if len(valid_rows) > 0:
    middle_rows = valid_rows[
        (valid_rows > valid_rows[0] + 20)
        & (valid_rows < valid_rows[-1] - 20)
    ]

    if len(middle_rows) > 0:
        body_width = np.median(row_widths[middle_rows])
    else:
        body_width = np.median(row_widths[valid_rows])

    strong_rows = valid_rows[
        row_widths[valid_rows] >= 0.55 * body_width
    ]

    if len(strong_rows) > 0:
        bottom_y = strong_rows[-1]
        mask[bottom_y + 1:, :] = 0


# =========================
# 7. 从 mask 提取外轮廓
# =========================
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

outline = np.zeros_like(mask)
cv2.drawContours(outline, contours, -1, 255, thickness=2)


# =========================
# 8. 把领口内部结构从原 edge 中拿回来
# =========================
control = outline.copy()

ys, xs = np.where(mask > 0)
if len(xs) > 0 and len(ys) > 0:
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    top_h = int(0.30 * (y2 - y1))
    neckline_region = np.zeros_like(mask)
    neckline_region[y1:y1 + top_h, x1:x2 + 1] = 255

    neckline_edges = cv2.bitwise_and(edge, neckline_region)
    control = cv2.bitwise_or(outline, neckline_edges)


# =========================
# 9. 保存结果
# =========================
cv2.imwrite(MASK_PATH, mask)
cv2.imwrite(OUTLINE_PATH, outline)
cv2.imwrite(CONTROL_PATH, control)

print("已生成：")
print("1)", MASK_PATH)
print("2)", OUTLINE_PATH)
print("3)", CONTROL_PATH)
