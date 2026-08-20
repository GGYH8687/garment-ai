import cv2
import numpy as np


OUTPUT_PATH = "inputs/garment_lineart.png"

canvas = np.zeros((512, 512), dtype=np.uint8)

# 连衣裙完整外轮廓：肩部、短袖、收腰侧缝与及膝直裙摆。
outline = np.array(
    [
        [230, 80],
        [198, 88],
        [168, 106],
        [176, 146],
        [202, 137],
        [214, 216],
        [226, 258],
        [211, 318],
        [214, 430],
        [298, 430],
        [301, 318],
        [286, 258],
        [298, 216],
        [310, 137],
        [336, 146],
        [344, 106],
        [314, 88],
        [282, 80],
        [256, 121],
        [230, 80],
    ],
    dtype=np.int32,
)

cv2.polylines(
    canvas,
    [outline],
    isClosed=True,
    color=255,
    thickness=3,
    lineType=cv2.LINE_AA,
)

# 简洁腰线，强化合体收腰结构；不添加任何面料纹理。
cv2.line(canvas, (226, 258), (286, 258), 255, 2, cv2.LINE_AA)

if not cv2.imwrite(OUTPUT_PATH, canvas):
    raise OSError(f"无法保存 {OUTPUT_PATH}")

print(f"已生成：{OUTPUT_PATH}")
