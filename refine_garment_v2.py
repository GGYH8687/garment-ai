import os

# 强制离线模式，避免 HF Hub 网络缓存检查（模型使用本地缓存）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


# =========================================================
# 1. 路径与参数
# =========================================================
IMAGE_PATH = "outputs/texture_warped_best.png"
MASK_PATH = "inputs/garment_mask_clean.png"

# Stage 4.1: 只改变 strength，其余全部固定
strength_list = [0.10, 0.15, 0.20, 0.25]


# =========================================================
# 2. 加载 Inpainting 管线（本地缓存）
# =========================================================
sd_snapshots = Path(
    ".cache/huggingface/hub/"
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots"
)
sd_model_path = next(path for path in sd_snapshots.iterdir() if path.is_dir())

print("Loading Stable Diffusion Inpainting pipeline (offline)...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    sd_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False,
).to("cuda")


# =========================================================
# 3. 读取图片
# =========================================================
image = Image.open(IMAGE_PATH).convert("RGB")
mask = Image.open(MASK_PATH).convert("RGB")


# =========================================================
# 4. Prompt
#
# Stage 4.1 暂时不要改
# 保持和原实验完全一致
# =========================================================
prompt = """
a realistic tailored women's sheath dress,
red and black plaid fabric,
V-neck, short sleeves, fitted waist, knee-length skirt,
preserve the plaid pattern and garment shape,
realistic fabric texture, subtle folds, seam details,
fashion photography
"""

negative_prompt = """
distorted pattern, blurry, low quality, extra accessories,
new garment design, different neckline, different sleeves,
text, watermark
"""


# =========================================================
# 5. 批量测试不同 strength
# =========================================================
for strength in strength_list:
    print()
    print("=" * 50)
    print("Testing strength:", strength)
    print("=" * 50)

    # 每一次都重新使用相同 seed
    # 保证实验尽量公平
    generator = torch.Generator(
        device="cuda"
    ).manual_seed(42)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=30,
        guidance_scale=6.5,
        strength=strength,
        generator=generator,
    ).images[0]

    # -----------------------------------------------------
    # 输出文件名
    # -----------------------------------------------------
    strength_name = int(strength * 100)
    output_path = (
        f"outputs/"
        f"refined_strength_{strength_name:02d}.png"
    )
    result.save(output_path)
    print("Saved:", output_path)

print()
print("All experiments finished.")
