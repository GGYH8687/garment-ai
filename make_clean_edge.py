import cv2


# 1. 读取原图
image = cv2.imread("inputs/garment.png")

if image is None:
    raise FileNotFoundError("无法读取 inputs/garment.png")

# 2. 转成灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 先进行高斯模糊
# 目的：抑制衣服表面的细小纹理
blurred = cv2.GaussianBlur(
    gray,
    (7, 7),
    0,
)

# 4. Canny 边缘检测
edges = cv2.Canny(
    blurred,
    threshold1=80,
    threshold2=160,
)

# 5. 保存
cv2.imwrite(
    "outputs/garment_edge_clean.png",
    edges,
)

print("已生成干净边缘图：outputs/garment_edge_clean.png")
