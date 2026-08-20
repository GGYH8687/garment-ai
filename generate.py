import torch
from diffusers import AutoPipelineForText2Image


print("正在加载模型...")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
)

pipe = pipe.to("cuda")

print("模型加载完成！")

prompt = """
a realistic studio product photograph of a women's fashion dress,
front view,
full garment visible,
elegant modern design,
high quality fabric,
white background,
professional fashion catalog photography,
highly detailed
"""

print("正在生成图片...")

image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

image.save("outputs/first_dress.png")

print("生成完成！")
print("图片保存在 outputs/first_dress.png")
