import os

# 强制离线模式，避免 HF Hub 网络缓存检查（所有模型均使用本地缓存）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline,
)


# ========= 1. 固定实验参数 =========
lineart_path = "inputs/garment_lineart.png"
fabric_files = [
    "inputs/fabrics/fabric_01.jpg",  # 红黑格纹
    "inputs/fabrics/fabric_02.jpg",  # 蓝白条纹
    "inputs/fabrics/fabric_03.jpg",  # 花卉图案
]

seed = 42
steps = 30
guidance_scale = 7.5
controlnet_scale = 1.0
strength_val = 0.60  # 已验证的平衡点：版型保持 + 布料覆盖
ip_scale = 0.60  # 已验证的最佳 IP-Adapter 强度

negative_prompt = """
person, model, mannequin, hanger, arms, legs, head,
text, watermark, complex background, blurry, low quality
"""


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


# ========= 3. 加载基础管线（SD v1.5 + ControlNet，本地缓存）=========
sd_snapshots = Path(
    ".cache/huggingface/hub/"
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots"
)
sd_model_path = next(path for path in sd_snapshots.iterdir() if path.is_dir())

print("Loading Stable Diffusion ControlNet pipeline...")
pipe_shape = StableDiffusionControlNetPipeline.from_pretrained(
    sd_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False,
).to("cuda")


# ========= 4. 阶段 A：生成基础版型图 =========
print("\n===== Stage A: 生成基础版型图 =====")
control_image = Image.open(lineart_path).convert("RGB")

prompt_shape = """
a fitted women's sheath dress,
V-neck,
short sleeves,
defined waist,
knee-length straight skirt,
front view,
standalone garment,
pure white background,
fashion catalog product image
"""

generator = torch.Generator(device="cuda").manual_seed(seed)

base = pipe_shape(
    prompt=prompt_shape,
    negative_prompt=negative_prompt,
    image=control_image,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    controlnet_conditioning_scale=controlnet_scale,
    generator=generator,
).images[0]

base.save("outputs/base_shape.png")
print("Saved: outputs/base_shape.png")


# ========= 5. 释放阶段 A 显存 =========
print("\n释放阶段 A 显存...")
del pipe_shape
del controlnet
torch.cuda.empty_cache()


# ========= 6. 阶段 B：加载 Inpainting + IP-Adapter =========
print("\n===== Stage B: 加载 Inpainting 管线 =====")
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    sd_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False,
)

# 使用本地缓存路径加载 IP-Adapter，离线加载
ip_adapter_snapshots = Path(
    ".cache/huggingface/hub/"
    "models--h94--IP-Adapter/snapshots"
)
ip_adapter_path = next(
    path for path in ip_adapter_snapshots.iterdir() if path.is_dir()
)

print("Loading IP-Adapter...")
pipe_inpaint.load_ip_adapter(
    str(ip_adapter_path),
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
)

pipe_inpaint = pipe_inpaint.to("cuda")


# ========= 7. 只在衣服区域做布料重绘（3 布料对照实验）=========
print("\n===== Stage B: 衣服区域布料重绘 (3 布料对照) =====")
base_image = Image.open("outputs/base_shape.png").convert("RGB")

# 统一使用膨胀 mask（5×5 核）
mask_file = "inputs/garment_mask_expanded.png"
mask_image = Image.open(mask_file).convert("RGB")

# 固定 IP-Adapter 强度
pipe_inpaint.set_ip_adapter_scale(ip_scale)

prompt_fabric = """
apply the referenced fabric to the entire dress,
including the bodice, sleeves, side panels, waist area, and skirt,
keep the same dress shape,
front view
"""

for fabric_path in fabric_files:
    fabric_stem = Path(fabric_path).stem
    print(f"\n--- fabric={fabric_stem} ---")
    fabric_image = Image.open(fabric_path).convert("RGB")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe_inpaint(
        prompt=prompt_fabric,
        negative_prompt=negative_prompt,
        image=base_image,
        mask_image=mask_image,
        ip_adapter_image=fabric_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        strength=strength_val,
        generator=generator,
    ).images[0]

    save_name = f"outputs/v2_mask_inpaint_{fabric_stem}.png"
    result.save(save_name)
    print("Saved:", save_name)

print("\nAll done!")
