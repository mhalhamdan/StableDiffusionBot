import torch
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from credentials import HUGGINGFACE_TOKEN
from PIL import Image
pretrained_model = "CompVis/stable-diffusion-v1-4"

print("Loading model...")
pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model, revision="fp16", torch_dtype=torch.float16, use_auth_token=HUGGINGFACE_TOKEN
    ).to("cuda")
print("Model loaded successfully!")

def generate(prompt, count) -> list[Image.Image]:
    all_images = [] 
    images = pipeline(prompt, num_images_per_prompt=count, num_inference_steps=50, guidance_scale=7.5).images
    all_images.extend(images)
    return all_images

if __name__ == "__main__":

    while True:
        prompt = input("Enter prompt: ")
        print("Generating...")
        all_images = generate(prompt, 1)
        print("Done:")
        print(all_images)
        print(type(all_images))
        all_images[0].show()