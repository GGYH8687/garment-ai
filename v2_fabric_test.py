from pathlib import Path

import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


# ========= 1. 加载 ControlNet =========
controlnet_snapshots = Path(
    ".cache/huggingface/hub/"
    "models--lllyasviel--sd-controlnet-canny/snapshots"
)
controlnet_model_path = next(
    path for path in controlnet_snapshots.iterdir() if path.is_dir()
)

controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
)


# ========= 2. 加载基础管线 =========
# 直接使用项目内已下载的 FP16 快照，避免重复下载基础模型。
sd_snapshots = Path(
    ".cache/huggingface/hub/"
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots"
)
sd_model_path = next(path for path in sd_snapshots.iterdir() if path.is_dir())

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False,
)


# ========= 3. 加载 IP-Adapter =========
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
)

pipe = pipe.to("cuda")

# IP-Adapter 强度
pipe.set_ip_adapter_scale(0.7)


# ========= 4. 读取输入 =========
control_image = Image.open("inputs/garment_lineart.png").convert("RGB")
fabric_image = Image.open("inputs/fabric.jpg").convert("RGB")

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

generator = torch.Generator(device="cuda").manual_seed(42)

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=control_image,
    ip_adapter_image=fabric_image,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    generator=generator,
)

image = result.images[0]
image.save("outputs/v2_fabric_test.png")

print("生成完成：outputs/v2_fabric_test.png")
