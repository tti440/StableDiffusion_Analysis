from typing import List
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from groundingdino.util.inference import load_model
from daam import trace
import matplotlib.pyplot as plt
import math
# if you want to use the model with double precision -> torch.float64 or torch.double
# if you want to use the model with single precision -> torch.float32 or torch.float
# if you want to use the model with half precision -> torch.float16 or torch.half
SEED = 1024	
def get_model(model_name: str, torch_dtype: torch.dtype = torch.float16):
	if model_name == "SD1.4":
		return StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch_dtype)
	elif model_name == "SD2.0":
		return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base", torch_dtype=torch_dtype)
	elif model_name == "SD2.1":
		return StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch_dtype)
	else:
		raise ValueError(f"Unknown model name: {model_name}")

def generate_img(pipe: StableDiffusionPipeline, text: str, num_samples: int = 1) -> List[Image]:
	generator = torch.Generator(pipe.device).manual_seed(SEED)
	prompts = [text] * num_samples
	images = []
	with torch.no_grad():
		for i, prompt in enumerate(prompts):
			image = pipe(prompt, generator=generator).images[0]
			images.append(image)
	return images


def generate_img_with_heatmap(pipe: StableDiffusionPipeline, text: str, nouns:[List[str]], num_samples: int = 1) -> List[Image]:
	generator = torch.Generator(pipe.device).manual_seed(SEED)
	prompts = [text] * num_samples
	images = []
	all_heatmaps=[]
	with torch.no_grad():
		for i, prompt in enumerate(prompts):
			with trace(pipe) as tc:
				output = pipe(prompt, generator=generator)
				image = output.images[0]
				images.append(image)
				global_heat_map = tc.compute_global_heat_map()
				heatmaps=[]
				for idx, word in enumerate(nouns):
					word_heat_map = global_heat_map.compute_word_heat_map(word)
					heatmaps.append(word_heat_map)
				all_heatmaps.append(heatmaps)
	return images, all_heatmaps

def display_heatmap(images, all_heatmaps: List, nouns:List[str]):
	for i, heatmaps in enumerate(all_heatmaps):
		plt.figure(figsize=(12, 8))
		if len(nouns) > 5:
			num_cols = 5
			num_rows = math.ceil(len(nouns) / num_cols)
		else:
			num_cols = len(nouns)
			num_rows = 1
		fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
		axs = axs.flatten()
		for j in range(len(nouns), len(axs)):
			axs[j].axis("off")
		for idx, noun in enumerate(nouns):
			# Apply overlay directly to the subplot
			heatmaps[idx].plot_overlay(images[i], ax=axs[idx])
			axs[idx].set_title(noun)
			axs[idx].axis("off")
		plt.savefig(f"heatmap_{i}.png")