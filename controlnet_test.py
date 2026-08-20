from pathlib import Path

import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


print("正在加载 ControlNet...")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
    use_safetensors=True,
)

print("正在加载 Stable Diffusion...")

# 直接使用项目内已下载的 FP16 快照，避免网络元数据不稳定。
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
)
pipe = pipe.to("cuda")

control_image = Image.open(
    "inputs/garment_lineart.png"
).convert("RGB")

prompt = """
a fitted women's sheath dress,
V-neck,
short sleeves,
defined waist,
knee-length straight skirt,
front view,
standalone garment,
pure white background,
fashion product photography
"""

negative_prompt = """
person,
woman,
mannequin,
hanger,
arms,
legs,
head,
sleeveless,
round neck,
loose dress,
oversized dress,
text,
watermark,
complex background
"""

generator = torch.Generator(device="cuda").manual_seed(42)

print("正在生成：outputs/controlnet_lineart.png")
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=control_image,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    generator=generator,
)

result.images[0].save("outputs/controlnet_lineart.png")
print("生成完成：outputs/controlnet_lineart.png")
