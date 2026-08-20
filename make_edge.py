import cv2


image = cv2.imread("inputs/garment.png")

if image is None:
    raise FileNotFoundError("无法读取 inputs/garment.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(
    gray,
    threshold1=100,
    threshold2=200,
)

cv2.imwrite("outputs/garment_edge.png", edges)

print("边缘图已生成：outputs/garment_edge.png")
