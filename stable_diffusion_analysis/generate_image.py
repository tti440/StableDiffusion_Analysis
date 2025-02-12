from collections import defaultdict
import os
from typing import List, Tuple
import torch
from models import generate_img, generate_img_with_heatmap, get_model
import spacy 
import pickle

if not os.path.exists("images"):
	os.makedirs("images")

NLP = spacy.load("en_core_web_sm")

def all_images(triples:List[Tuple[str,str,str]], model_names:List[str] = ["SD1.4", "SD2.0", "SD2.1"], torch_dtype:torch.dtype = torch.float16, num_samples:int = 1, with_heatmap:bool = True):
	for model_name in model_names:
		generate_images(triples, model_name, torch_dtype, num_samples, with_heatmap)

def generate_images(triples:List[Tuple[str,str,str]], model_name:str, torch_dtype:torch.dtype = torch.float16, num_samples:int = 1, pipe = None, with_heatmap:bool = True):
	if pipe is None:
		pipe = get_model(model_name, torch_dtype)
	pipe.to("cuda")
	if not os.path.exists(f"images/{model_name}"):
		os.makedirs(f"images/{model_name}")	
	if with_heatmap:
		with torch.no_grad():
			heatmaps = defaultdict(dict)
			for i, triple in enumerate(triples):
				neutral, feminine, masculine = triple
				doc = NLP(feminine)
				femi_nouns = [token.text for token in doc if token.pos_ == "NOUN"]
				doc = NLP(masculine)
				masc_nouns = [token.text for token in doc if token.pos_ == "NOUN"]
				neutral_images = generate_img(pipe, neutral, num_samples=num_samples)
				feminine_images, femi_heatmaps = generate_img_with_heatmap(pipe, feminine, femi_nouns, num_samples=num_samples)
				masculine_images, masc_heatmaps = generate_img_with_heatmap(pipe, masculine, masc_nouns, num_samples=num_samples)
				heatmaps[i] = {
					"feminine": femi_heatmaps,
					"masculine": masc_heatmaps
				}
				if not os.path.exists(f"images/{model_name}/neutral"):
					os.makedirs(f"images/{model_name}/neutral")
				if not os.path.exists(f"images/{model_name}/feminine"):
					os.makedirs(f"images/{model_name}/feminine")
				if not os.path.exists(f"images/{model_name}/masculine"):
					os.makedirs(f"images/{model_name}/masculine")
				
				for j, img in enumerate(neutral_images):
					if not os.path.exists(f"images/{model_name}/neutral/{i}"):
						os.makedirs(f"images/{model_name}/neutral/{i}")
					img.save(f"images/{model_name}/neutral/{i}/{j}.png")
				for j, img in enumerate(feminine_images):
					if not os.path.exists(f"images/{model_name}/feminine/{i}"):
						os.makedirs(f"images/{model_name}/feminine/{i}")
					img.save(f"images/{model_name}/feminine/{i}/{j}.png")
				for j, img in enumerate(masculine_images):
					if not os.path.exists(f"images/{model_name}/masculine/{i}"):
						os.makedirs(f"images/{model_name}/masculine/{i}")
					img.save(f"images/{model_name}/masculine/{i}/{j}.png")
		with open(f"heatmap_{model_name}.pkl", "wb") as f:
			pickle.dump(heatmaps, f)
	else:
		with torch.no_grad():
			for i, triple in enumerate(triples):
				neutral, feminine, masculine = triple
				neutral_images = generate_img(pipe, neutral, num_samples=num_samples)
				feminine_images = generate_img(pipe, feminine, num_samples=num_samples)
				masculine_images = generate_img(pipe, masculine, num_samples=num_samples)
				if not os.path.exists(f"images/{model_name}/neutral"):
					os.makedirs(f"images/{model_name}/neutral")
				if not os.path.exists(f"images/{model_name}/feminine"):
					os.makedirs(f"images/{model_name}/feminine")
				if not os.path.exists(f"images/{model_name}/masculine"):
					os.makedirs(f"images/{model_name}/masculine")
				
				for j, img in enumerate(neutral_images):
					if not os.path.exists(f"images/{model_name}/neutral/{i}"):
						os.makedirs(f"images/{model_name}/neutral/{i}")
					img.save(f"images/{model_name}/neutral/{i}.png")
				for j, img in enumerate(feminine_images):
					if not os.path.exists(f"images/{model_name}/feminine/{i}"):
						os.makedirs(f"images/{model_name}/feminine/{i}")
					img.save(f"images/{model_name}/feminine/{i}/{j}.png")
				for j, img in enumerate(masculine_images):
					if not os.path.exists(f"images/{model_name}/masculine/{i}"):
						os.makedirs(f"images/{model_name}/masculine/{i}")
					img.save(f"images/{model_name}/masculine/{i}/{j}.png")