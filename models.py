import torch
from diffusers import StableDiffusionPipeline

# if you want to use the model with double precision -> torch.float64 or torch.double
# if you want to use the model with single precision -> torch.float32 or torch.float
# if you want to use the model with half precision -> torch.float16 or torch.half
        
def get_model(model_name: str, torch_dtype: torch.dtype = torch.float16):
    if model_name == "SD1.4":
        return StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch_dtype)
    elif model_name == "SD2.0":
        return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base", torch_dtype=torch_dtype)
    elif model_name == "SD2.1":
        return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

  