import PIL
import torch
from pipeline_onnx_stable_diffusion_instruct_pix2pix import OnnxStableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
import requests
pipe = OnnxStableDiffusionInstructPix2PixPipeline.from_pretrained("./ip2p_onnx", provider="DmlExecutionProvider", safety_checker=None)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
image = download_image(url)
prompt = "turn him into cyborg"
seed=42
generator = torch.Generator()
generator = generator.manual_seed(seed)
images = pipe(prompt, image=image, num_inference_steps=10, guidance_scale=7.5, image_guidance_scale=1, generator=generator).images[0]
images.save("cyborg.png")