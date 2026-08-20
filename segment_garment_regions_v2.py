import cv2
import numpy as np
import os


OUTPUT_DIR = "outputs/"


def load_binary(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return np.where(img > 127, 255, 0).astype(np.uint8)


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)


    if num_labels <= 1:
        return mask


    # 跳过 0 号背景，找面积最大的前景连通域
    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]


    for i in range(2, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i


    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result



# -----------------------------------------------------
# 1. 读取 Stage 5.1 结果
# -----------------------------------------------------


upper = load_binary(os.path.join(OUTPUT_DIR, "mask_upper.png"))
left_sleeve = load_binary(os.path.join(OUTPUT_DIR, "mask_left_sleeve.png"))
right_sleeve = load_binary(os.path.join(OUTPUT_DIR, "mask_right_sleeve.png"))
skirt = load_binary(os.path.join(OUTPUT_DIR, "mask_skirt.png"))


full_mask = load_binary("inputs/garment_mask_clean.png")



# -----------------------------------------------------
# 2. 清理左右袖，只保留最大连通区域
# -----------------------------------------------------


left_sleeve_clean = keep_largest_component(left_sleeve)
right_sleeve_clean = keep_largest_component(right_sleeve)



# -----------------------------------------------------
# 3. 强制区域互斥
# 优先级：
# sleeve > upper > skirt
# -----------------------------------------------------


# 先把 upper 中与 sleeve 重叠的部分去掉
upper_clean = upper.copy()
upper_clean[left_sleeve_clean > 0] = 0
upper_clean[right_sleeve_clean > 0] = 0


# 再把 skirt 中与 upper/sleeve 重叠的部分去掉
skirt_clean = skirt.copy()
skirt_clean[upper_clean > 0] = 0
skirt_clean[left_sleeve_clean > 0] = 0
skirt_clean[right_sleeve_clean > 0] = 0



# -----------------------------------------------------
# 4. 可选：再做一次与 full mask 相交，防止超界
# -----------------------------------------------------


upper_clean[full_mask == 0] = 0
left_sleeve_clean[full_mask == 0] = 0
right_sleeve_clean[full_mask == 0] = 0
skirt_clean[full_mask == 0] = 0



# -----------------------------------------------------
# 5. 保存
# -----------------------------------------------------


cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_upper_clean.png"), upper_clean)
cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_left_sleeve_clean.png"), left_sleeve_clean)
cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_right_sleeve_clean.png"), right_sleeve_clean)
cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_skirt_clean.png"), skirt_clean)


print("Stage 5.1.1 finished.")
