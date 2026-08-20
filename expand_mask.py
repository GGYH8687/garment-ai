import cv2

mask = cv2.imread("inputs/garment_mask.png", cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
expanded = cv2.dilate(mask, kernel, iterations=1)

cv2.imwrite("inputs/garment_mask_expanded.png", expanded)
print("Saved: inputs/garment_mask_expanded.png")
