import os

# 强制离线模式，避免 HF Hub 网络缓存检查（模型使用本地缓存）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


IMAGE_PATH = "outputs/texture_warped_best.png"
MASK_PATH = "inputs/garment_mask_clean.png"
OUTPUT_PATH = "outputs/refined_garment.png"


# ========= 加载 Inpainting 管线（本地缓存）=========
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


image = Image.open(IMAGE_PATH).convert("RGB")
mask = Image.open(MASK_PATH).convert("RGB")


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


generator = torch.Generator(device="cuda").manual_seed(42)


result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=30,
    guidance_scale=6.5,
    strength=0.30,
    generator=generator
).images[0]


result.save(OUTPUT_PATH)
print("Saved:", OUTPUT_PATH)
