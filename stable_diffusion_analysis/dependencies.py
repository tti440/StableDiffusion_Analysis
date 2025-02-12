from typing import List
from detect_objects import bias_score
import pickle
import spacy
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
NLP = spacy.load("en_core_web_sm")
THRESHOLD = 0.8

def all_dependencies(model_names:List[str] = ["SD1.4", "SD2.0", "SD2.1"], num_samples:int = 1, threshold:float = 0.8):
	dependencies = {}
	object_guidance = {}
	for model_name in model_names:
		dependencies[model_name], object_guidance[model_name] = image_dependencies(model_name, num_samples, threshold)
	return dependencies, object_guidance
def image_dependencies(model_name: str, num_samples: int = 1, threshold: float = 0.8):
	heatmap = pickle.load(open(f"heatmap_{model_name}.pkl", "rb"))
	res = pickle.load(open(f"detect_object_{model_name}_results.pkl", "rb"))
	dependencies = {
		'explicit_guided': defaultdict(int),
  		'implicit_guided': defaultdict(int),
		'explicit_independent': defaultdict(int),
		'implicit_independent': defaultdict(int),
		'hidden': defaultdict(int),
		'femi': {
			'explicit_guided': defaultdict(int),
  			'implicit_guided': defaultdict(int),
			'explicit_independent': defaultdict(int),
			'implicit_independent': defaultdict(int),
		},
		'masc': {
			'explicit_guided': defaultdict(int),
  			'implicit_guided': defaultdict(int),
			'explicit_independent': defaultdict(int),
			'implicit_independent': defaultdict(int),
		}
	}
	object_guidance = defaultdict(list[str])
	num_prompts = len(res)
	print(f"Start analysing dependencies for {model_name}")
	for i in tqdm(range(num_prompts), desc="Processing Prompts: "):
		data = res[str(i)]
		femi_objects = data["femi_objects"]
		masc_objects = data["masc_objects"]
		neutral_objects = data["neutral_objects"]
		femi_masks = data["femi_masks"]
		masc_masks = data["masc_masks"]
		femi_prompt = data["femi_prompt"]
		masc_prompt = data["masc_prompt"]
		femi_heatmaps = heatmap[i]["feminine"]
		masc_heatmaps = heatmap[i]["masculine"]
		doc = NLP(femi_prompt)
		femi_nouns = [token.text for token in doc if token.pos_ == "NOUN"]
		doc = NLP(masc_prompt)
		masc_nouns = [token.text for token in doc if token.pos_ == "NOUN"]
		for j in range(num_samples):
			femi_heatmap = femi_heatmaps[j]
			femi_heatmap = resize_heatmap(femi_heatmap)
			masc_heatmap = masc_heatmaps[j]
			masc_heatmap = resize_heatmap(masc_heatmap)
			set_objects_femi = femi_objects[j]
			set_objects_masc = masc_objects[j]
			# I am not sure if the cooccurence calculation
			set_objects_neutral = neutral_objects[j]
			femi_mask = femi_masks[j]
			masc_mask = masc_masks[j]
			if len(femi_mask.shape) == 3:
				femi_mask = np.expand_dims(femi_mask, axis=0)
			if len(masc_mask.shape) == 3:
				masc_mask = np.expand_dims(masc_mask, axis=0)
			for index, object in enumerate(set_objects_femi):
				mask = femi_mask[index]
				guided, guiding_word = guided_judge(mask, femi_heatmap, femi_nouns, threshold)
				if guiding_word is not None:
					object_guidance[object].append(guiding_word)
				if object in femi_nouns:
					if guided:
						if object in dependencies['explicit_guided']:
							dependencies['explicit_guided'][object] += 1
						else:
							dependencies['explicit_guided'][object] = 1
						if object in dependencies['femi']['explicit_guided']:
							dependencies['femi']['explicit_guided'][object] += 1
						else:
							dependencies['femi']['explicit_guided'][object] = 1
						dependencies['masc']['explicit_guided'][object] = 0
	  
					else:
						if object in dependencies['explicit_independent']:
							dependencies['explicit_independent'][object] += 1
						else:
							dependencies['explicit_independent'][object] = 1
						if object in dependencies['femi']['explicit_independent']:
							dependencies['femi']['explicit_independent'][object] += 1
						else:
							dependencies['femi']['explicit_independent'][object] = 1	
						dependencies['masc']['explicit_independent'][object] = 0
				else:
					if guided:
						if object in dependencies['implicit_guided']:
							dependencies['implicit_guided'][object] += 1
						else:
							dependencies['implicit_guided'][object] = 1
						if object in dependencies['femi']['implicit_guided']:
							dependencies['femi']['implicit_guided'][object] += 1
						else:
							dependencies['femi']['implicit_guided'][object] = 1
						dependencies['masc']['implicit_guided'][object] = 0
					else:
						if object in dependencies['implicit_independent']:
							dependencies['implicit_independent'][object] += 1
						else:
							dependencies['implicit_independent'][object] = 1
						if object in dependencies['femi']['implicit_independent']:
							dependencies['femi']['implicit_independent'][object] += 1
						else:
							dependencies['femi']['implicit_independent'][object] = 1
						dependencies['masc']['implicit_independent'][object] = 0
	  
				for noun in femi_nouns:
					if noun not in set_objects_femi:
						if noun in dependencies['hidden']:
							dependencies['hidden'][noun] += 1
						else:
							dependencies['hidden'][noun] = 1
	   
			for index, object in enumerate(set_objects_masc):
				mask = masc_mask[index]
				guided, guiding_word = guided_judge(mask, masc_heatmap, masc_nouns, threshold)
				object_guidance[object].append(guiding_word)
				if object in masc_nouns:
					if guided:
						if object in dependencies['explicit_guided']:
							dependencies['explicit_guided'][object] += 1
						else:
							dependencies['explicit_guided'][object] = 1
						if object in dependencies['masc']['explicit_guided']:
							dependencies['masc']['explicit_guided'][object] += 1
						else:
							dependencies['masc']['explicit_guided'][object] = 1
						dependencies['femi']['explicit_guided'][object] = 0
					else:
						if object in dependencies['explicit_independent']:
							dependencies['explicit_independent'][object] += 1
						else:
							dependencies['explicit_independent'][object] = 1
						if object in dependencies['masc']['explicit_independent']:
							dependencies['masc']['explicit_independent'][object] += 1
						else:
							dependencies['masc']['explicit_independent'][object] = 1
						dependencies['femi']['explicit_independent'][object] = 0
				else:
					if guided:
						if object in dependencies['implicit_guided']:
							dependencies['implicit_guided'][object] += 1
						else:
							dependencies['implicit_guided'][object] = 1
						if object in dependencies['masc']['implicit_guided']:
							dependencies['masc']['implicit_guided'][object] += 1
						else:
							dependencies['masc']['implicit_guided'][object] = 1
						dependencies['femi']['implicit_guided'][object] = 0
					else:
						if object in dependencies['implicit_independent']:
							dependencies['implicit_independent'][object] += 1
						else:
							dependencies['implicit_independent'][object] = 1
						if object in dependencies['masc']['implicit_independent']:
							dependencies['masc']['implicit_independent'][object] += 1
						else:
							dependencies['masc']['implicit_independent'][object] = 1
						dependencies['femi']['implicit_independent'][object] = 0
	  
				for noun in masc_nouns:
					if noun not in set_objects_masc:
						if noun in dependencies['hidden']:
							dependencies['hidden'][noun] += 1
						else:
							dependencies['hidden'][noun] = 1
	return dependencies, object_guidance

def guided_judge(mask:np.ndarray, heatmaps:List[np.ndarray], nouns:List[str], threshold:float):
	guided = False
	largest_coverage = float('-inf')
	index = -1
	for i, heatmap in enumerate(heatmaps):
		heatmap = heatmap.cpu().detach().numpy()
		heatmap = np.where(heatmap > 0.6, 1, 0)
		overlap = np.logical_and(mask, heatmap)
		coverage = np.count_nonzero(overlap) / np.count_nonzero(mask)
		if coverage >= threshold:
			guided = True
			if coverage > largest_coverage:
				largest_coverage = coverage
				index = i	
	if index == -1:
		return guided, None
	return guided, nouns[index]

def bias_score_dependencies(results):
	all_scores = {}
	for model, dependencies in results.items():
		scores = {
			'implicit_guided': 0,
			'implicit_independent': 0,
		}
		femi_implicit_guided = dependencies['femi']['implicit_guided']
		femi_implicit_independent = dependencies['femi']['implicit_independent']
		masc_implicit_guided = dependencies['masc']['implicit_guided']
		masc_implicit_independent = dependencies['masc']['implicit_independent']
		implicit_guided = bias_score(femi_implicit_guided, masc_implicit_guided)
		implicit_independent = bias_score(femi_implicit_independent, masc_implicit_independent)
		scores['implicit_guided'] = implicit_guided
		scores['implicit_independent'] = implicit_independent
		all_scores[model] = scores
	return all_scores

def resize_heatmap(heatmaps:List[np.ndarray]):
	resized=[]
	for heatmap in heatmaps:
		im = heatmap.heatmap.unsqueeze(0).unsqueeze(0)
		im = F.interpolate(im.float().detach(), size=(512, 512), mode='bicubic')
		im = im.cpu().detach().squeeze(0)
		resized.append(im)
	return resized