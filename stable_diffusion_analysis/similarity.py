from typing import Tuple, List
from models import get_model
from model_utils import get_text_embedding, get_image_denoising, calculate_cosine_similarity, get_ssim_score, get_diffpix, get_resnet_similarity, get_clip_similarity, get_dino_similarity, split_product
import torch
import os
from collections import defaultdict
from transformers import CLIPImageProcessor, CLIPModel
from torchvision import models
import spacy
from generate_image import generate_images
from PIL import Image

RESNET = models.resnet50(pretrained=True)
CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to("cuda")
CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
DINO = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
NLP = spacy.load("en_core_web_sm")

def all_similarities(triples: List[Tuple[str, str, str]], model_names:List[str] = ["SD1.4", "SD2.0", "SD2.1"], torch_dtype: torch.dtype = torch.float16, num_samples: int = 1, generate:bool = False, with_heatmap:bool = True):
	similarities = {}
	for model_name in model_names:
		print(model_name)
		similarities[model_name] = calc_similarities(triples, model_name, torch_dtype, num_samples, generate, with_heatmap)
     
	return similarities

def calc_similarities(triples: List[Tuple[str, str, str]], model_name:str, torch_dtype: torch.dtype, num_samples: int, generate:bool, with_heatmap:bool):
	pipe = get_model(model_name, torch_dtype)
	pipe.to("cuda")
	if not os.path.exists(f"images/{model_name}") or generate:
		generate_images(triples, model_name, torch_dtype, num_samples, pipe=pipe, with_heatmap=with_heatmap)
	path = f"images/{model_name}"
	with torch.no_grad():
		similarities = defaultdict(dict)
		for i, triple in enumerate(triples):
			neutral, feminine, masculine = triple
			# get text embeddings
			neutral_textembedding = get_text_embedding(neutral, pipe)
			feminine_textembedding = get_text_embedding(feminine, pipe)
			masculine_textembedding = get_text_embedding(masculine, pipe)
	
			#get denosing
			neutral_denosing = get_image_denoising(neutral, pipe)
			feminine_denosing = get_image_denoising(feminine, pipe)
			masculine_denosing = get_image_denoising(masculine, pipe)

			#image generation
			neutral_images = []
			feminine_images = []
			masculine_images = []
			prompt_path = os.path.join(f"{path}/neutral", str(i))
			for image in os.listdir(prompt_path):
				image_path = os.path.join(prompt_path, image)
				neutral_images.append(Image.open(image_path))
			prompt_path = os.path.join(f"{path}/feminine", str(i))
			for image in os.listdir(prompt_path):
				image_path = os.path.join(prompt_path, image)
				feminine_images.append(Image.open(image_path))
			prompt_path = os.path.join(f"{path}/masculine", str(i))
			for image in os.listdir(prompt_path):
				image_path = os.path.join(prompt_path, image)
				masculine_images.append(Image.open(image_path))

			#similarity
			prompt_sim_nf = calculate_cosine_similarity(neutral_textembedding, feminine_textembedding)
			prompt_sim_nm = calculate_cosine_similarity(neutral_textembedding, masculine_textembedding)
			z0_sim_nf = calculate_cosine_similarity(neutral_denosing, feminine_denosing)
			z0_sim_nm = calculate_cosine_similarity(neutral_denosing, masculine_denosing)
	
			#compute ssim score
			total_sim_in_prompt = {}
			for j in range(num_samples):
				neutral_image = neutral_images[j]
				feminine_image = feminine_images[j]
				masculine_image = masculine_images[j]
				ssim_score_nf, nf_map = get_ssim_score(neutral_image, feminine_image)
				ssim_score_nm, nm_map = get_ssim_score(neutral_image, masculine_image)
	
				#diff. pix
				diff_pix_nf = get_diffpix(neutral_image, feminine_image, nf_map)
				diff_pix_nm = get_diffpix(neutral_image, masculine_image, nm_map)
	
				#3 models
				resnet_nf = get_resnet_similarity(neutral_image, feminine_image, RESNET)
				resnet_nm = get_resnet_similarity(neutral_image, masculine_image, RESNET)
				clip_nf = get_clip_similarity(neutral_image, feminine_image, CLIP_MODEL, CLIP_PROCESSOR)
				clip_nm = get_clip_similarity(neutral_image, masculine_image, CLIP_MODEL, CLIP_PROCESSOR)
				dino_nf = get_dino_similarity(neutral_image, feminine_image, DINO)
				dino_nm = get_dino_similarity(neutral_image, masculine_image, DINO)
	
				#split product
				sp_nf = split_product(neutral_image, feminine_image)
				sp_nm = split_product(neutral_image, masculine_image)
				total_sim_in_prompt[j] = {
					"ssim_score_nf": ssim_score_nf,
					"ssim_score_nm": ssim_score_nm,
					"diff_pix_nf": diff_pix_nf,
					"diff_pix_nm": diff_pix_nm,
					"resnet_nf": resnet_nf,
					"resnet_nm": resnet_nm,
					"clip_nf": clip_nf,
					"clip_nm": clip_nm,
					"dino_nf": dino_nf,
					"dino_nm": dino_nm,
					"sp_nf": sp_nf,
					"sp_nm": sp_nm
				}
			similarities[i] = {
				"prompt_sim_nf": prompt_sim_nf,
				"prompt_sim_nm": prompt_sim_nm,
				"z0_sim_nf": z0_sim_nf,
				"z0_sim_nm": z0_sim_nm,
				"ssim_score_nf": sum([instance["ssim_score_nf"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"ssim_score_nm": sum([instance["ssim_score_nm"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"diff_pix_nf": sum([instance["diff_pix_nf"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"diff_pix_nm": sum([instance["diff_pix_nm"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"resnet_nf": sum([instance["resnet_nf"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"resnet_nm": sum([instance["resnet_nm"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"clip_nf": sum([instance["clip_nf"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"clip_nm": sum([instance["clip_nm"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"dino_nf": sum([instance["dino_nf"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"dino_nm": sum([instance["dino_nm"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"sp_nf": sum([instance["sp_nf"] for instance in total_sim_in_prompt.values()]) / num_samples,
				"sp_nm": sum([instance["sp_nm"] for instance in total_sim_in_prompt.values()]) / num_samples
			}
				
	sum_similarities = {}
	for key in similarities[0].keys():
		sum_similarities[key] = sum([similarity[key] for similarity in similarities.values()]) / len(similarities)
	return sum_similarities