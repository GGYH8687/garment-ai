import os

# 强制离线模式，避免 HF Hub 网络缓存检查（所有模型均使用本地缓存）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


# ========= 1. 固定实验参数 =========
LINEART_PATH = "inputs/garment_lineart.png"
FABRIC_DIR = "inputs/fabrics"
OUTPUT_DIR = "outputs/v2_batch"

# 3 张布料图（差异明显：格纹 / 条纹 / 花卉）
fabric_files = ["fabric_01.jpg", "fabric_02.jpg", "fabric_03.jpg"]

# 要测试的 4 个 IP-Adapter scale
ip_scales = [0.0, 0.3, 0.6, 0.9]

# 固定不变的生成参数
seed = 42
num_inference_steps = 30
guidance_scale = 7.5
controlnet_scale = 1.0

# 固定的提示词
prompt = """
a fitted women's sheath dress,
V-neck,
short sleeves,
defined waist,
knee-length straight skirt,
front view,
standalone garment,
pure white background,
fashion catalog product image,
made of the referenced fabric
"""

negative_prompt = """
person,
model,
mannequin,
hanger,
arms,
legs,
head,
text,
watermark,
complex background,
blurry,
low quality
"""

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ========= 2. 加载 ControlNet =========
controlnet_snapshots = Path(
    ".cache/huggingface/hub/"
    "models--lllyasviel--sd-controlnet-canny/snapshots"
)
controlnet_model_path = next(
    path for path in controlnet_snapshots.iterdir() if path.is_dir()
)

print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
)


# ========= 3. 加载基础管线 =========
sd_snapshots = Path(
    ".cache/huggingface/hub/"
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots"
)
sd_model_path = next(path for path in sd_snapshots.iterdir() if path.is_dir())

print("Loading Stable Diffusion pipeline...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False,
)


# ========= 4. 加载 IP-Adapter（使用本地缓存路径，离线加载）=========
ip_adapter_snapshots = Path(
    ".cache/huggingface/hub/"
    "models--h94--IP-Adapter/snapshots"
)
ip_adapter_path = next(
    path for path in ip_adapter_snapshots.iterdir() if path.is_dir()
)

print("Loading IP-Adapter...")
pipe.load_ip_adapter(
    str(ip_adapter_path),
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
)

pipe = pipe.to("cuda")


# ========= 5. 读取固定的版型控制图 =========
control_image = Image.open(LINEART_PATH).convert("RGB")


# ========= 6. 开始批量实验：3 布料 × 4 scale = 12 张 =========
for fabric_name in fabric_files:
    fabric_path = Path(FABRIC_DIR) / fabric_name
    if not fabric_path.exists():
        print(f"Skip (not found): {fabric_path}")
        continue

    fabric_image = Image.open(fabric_path).convert("RGB")
    fabric_stem = fabric_path.stem

    for scale in ip_scales:
        print(f"Generating: {fabric_stem}, IP scale = {scale}")

        # 每次都重新设定 scale
        pipe.set_ip_adapter_scale(scale)

        # 每次都重新创建 generator，保证 seed 真正固定
        generator = torch.Generator(device="cuda").manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,              # ControlNet：版型
            ip_adapter_image=fabric_image,    # IP-Adapter：布料
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator,
        )

        image = result.images[0]
        save_name = f"{fabric_stem}_ip{scale:.1f}.png"
        save_path = Path(OUTPUT_DIR) / save_name
        image.save(save_path)
        print("Saved:", save_path)

print("All done!")
