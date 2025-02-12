from typing import Tuple, List, Union
from model_utils import object_detections, load_dino, load_sam2, load_ram
from PIL import Image
import torch
import os
from collections import defaultdict
import torch.nn.functional as F
from scipy.stats import chi2_contingency
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
_SAM2 = None
_RAM = None
_RAM_TRANSFORM = None
_DINO = None

def get_sam2():
	global _SAM2
	if _SAM2 is None:
		_SAM2 = load_sam2()
	return _SAM2

def get_ram():
	global _RAM, _RAM_TRANSFORM
	if _RAM is None or _RAM_TRANSFORM is None:
		_RAM, _RAM_TRANSFORM = load_ram()
	return _RAM, _RAM_TRANSFORM

def get_dino():
	global _DINO
	if _DINO is None:
		_DINO = load_dino()
	return _DINO

def all_detect_objects(triples: List[Tuple[str, str, str]], model_names:List[str] = ["SD1.4", "SD2.0", "SD2.1"]):
	combined_results = {}
	for model_name in model_names:
		combined_results[model_name] = detect_generated_objects(triples, model_name)
	return combined_results

def detect_generated_objects(triples:List[Tuple[str,str,str]], model_name:str):
	SAM2 = get_sam2()
	RAM, RAM_TRANSFORM = get_ram()
	DINO = get_dino()
	feminine = []
	masculine = []
	for triple in triples:
		feminine.append(triple[1])
		masculine.append(triple[2])
	
	#feminine
	combined_results= defaultdict(dict)
	root_path = os.getcwd()
	neutral_path = os.path.join(root_path, f"images/{model_name}/neutral")
	femi_path = os.path.join(root_path, f"images/{model_name}/feminine")
	masc_path = os.path.join(root_path, f"images/{model_name}/masculine")
	print(f"Start detecting objects for {model_name}")
	for prompt_index in tqdm(os.listdir(femi_path), desc="Processing Prompt Dirs: "):
		prompt_path = os.path.join(femi_path, prompt_index)
		images = []
		combined_results[prompt_index] = {
			"femi_prompt": feminine[int(prompt_index)],
			"masc_prompt": masculine[int(prompt_index)],
		}
		print(f"Processing Prompt {prompt_index} for female")
		for image in os.listdir(prompt_path):
			image_path = os.path.join(prompt_path, image)
			image = Image.open(image_path)
			images.append(image)
		rgs_results_femi = object_detections(images, SAM2, RAM_TRANSFORM, RAM, DINO)
		femi_objects = rgs_results_femi["nouns"]
		femi_masks = rgs_results_femi["masks"]
		prompt_path = os.path.join(masc_path, prompt_index)
		images = []
		print(f"Processing Prompt {prompt_index} for masculine")
		for image in os.listdir(prompt_path):
			image_path = os.path.join(prompt_path, image)
			image = Image.open(image_path)
			images.append(image)
		rgs_results_masc = object_detections(images, SAM2, RAM_TRANSFORM, RAM, DINO)
		masc_objects = rgs_results_masc["nouns"]
		masc_masks = rgs_results_masc["masks"]
		prompt_path = os.path.join(neutral_path, prompt_index)
		images = []
		print(f"Processing Prompt {prompt_index} for neutral")
		for image in os.listdir(prompt_path):
			image_path = os.path.join(prompt_path, image)
			image = Image.open(image_path)
			images.append(image)
		rgs_results_neutral = object_detections(images, SAM2, RAM_TRANSFORM, RAM, DINO)
		neutral_objects = rgs_results_neutral["nouns"]
  
		cos_sim_nf = compute_co_occurrence_similarity(neutral_objects, femi_objects)
		cos_sim_nm = compute_co_occurrence_similarity(neutral_objects, masc_objects)
		chi2, p_value = chi_square_test(femi_objects, masc_objects)
		bias_scores = bias_score(masc_objects, femi_objects)
		combined_results[prompt_index]["cos_sim_objects"] = {
			"neutral_feminine": cos_sim_nf,
			"neutral_masculine": cos_sim_nm
		}
		combined_results[prompt_index]["femi_objects"] = femi_objects
		combined_results[prompt_index]["masc_objects"] = masc_objects
		combined_results[prompt_index]["neutral_objects"] = neutral_objects
		combined_results[prompt_index]["chi_square"] = [chi2, p_value]
		combined_results[prompt_index]["femi_masks"] = femi_masks
		combined_results[prompt_index]["masc_masks"] = masc_masks
		combined_results[prompt_index]["bias_scores"] = bias_scores
	with open(f"detect_object_{model_name}_results.pkl", "wb") as f:
		pickle.dump(combined_results, f)
	return combined_results
	
   

def count_objects(nouns1: List[str], nouns2: List[str]):
	femi_count = defaultdict(int)
	masc_count = defaultdict(int)
	for nouns in nouns1:
		for noun in nouns:
			if noun in femi_count:
				femi_count[noun] += 1
			else:
				femi_count[noun] = 1
			masc_count[noun] = 0
	for nouns in nouns2:
		for noun in nouns:
			if noun in masc_count:
				masc_count[noun] += 1
			else:
				masc_count[noun] = 1
			femi_count[noun] = 0
	return femi_count, masc_count

def compute_co_occurrence_similarity(nouns1: List[str], nouns2: List[str]):
	counts1, counts2 = count_objects(nouns1, nouns2)
	tensor1 = torch.tensor(list(counts1.values()), dtype=torch.float32)
	tensor2 = torch.tensor(list(counts2.values()), dtype=torch.float32)
	tensor1 = F.normalize(tensor1, p=2, dim=0)
	tensor2 = F.normalize(tensor2, p=2, dim=0)
	return F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()

def chi_square_test(nouns1: List[str], nouns2: List[str]):
	counts1, counts2 = count_objects(nouns1, nouns2)
	contingency_table = [list(counts1.values()), list(counts2.values())]
	chi2, p_value, _, _ = chi2_contingency(contingency_table)
	return chi2, p_value

def bias_score(nouns1: Union[List[str] | dict[str,int]], nouns2: Union[List[str] | dict[str,int]]):
	if isinstance(nouns1, list):
		nouns1, nouns2 = count_objects(nouns1, nouns2)
	bias_score = defaultdict(float)
	for noun in nouns1:
		bias_score[noun] = nouns1[noun] / (nouns1[noun] + nouns2[noun])
	return bias_score
	
def get_figure_cooccurrence(results:dict):
	significant_objects = []
	cos_sim_nf = 0
	cos_sim_nm = 0
	bias_total = defaultdict(list[float, int])
	all_results = defaultdict(dict)
	for model, result in results.items():
		for prompt_idx, value in result.items():
			femi_object = value["femi_objects"]
			masc_object = value["masc_objects"]
			chi2, p_value = value["chi_square"]
			bias_score = value["bias_scores"]
			cos_sim = value["cos_sim_objects"]
			cos_sim_nf += cos_sim['neutral_feminine']
			cos_sim_nm += cos_sim['neutral_masculine']
			if p_value < 0.05:
				significant_objects.append({
					"prompt_idx": prompt_idx,
					"femi_objects": femi_object,
					"masc_objects": masc_object,
					"chi_square": chi2,
					"p_value": p_value,
				})
			for label, score in bias_score.items():
				if label not in bias_total:
					bias_total[label] = [0.0, 0]
				bias_total[label][0] += score
				bias_total[label][1] += 1

		bias_score = defaultdict(float)
		for label, value in bias_total.items():
			bias_score[label] = value[0] / value[1]
		bias_score = sorted(bias_score.items(), key=lambda x: x[1], reverse=True)
		#extract top10 and bottom 10 in bar chart
		#show the score of bias in bar chart
		# Adding "..." separator with a very small height
		top_labels = [label for label, score in bias_score[:10]]
		top_scores = [score for label, score in bias_score[:10]]
		bottom_labels = [label for label, score in bias_score[-10:][::-1]]
		bottom_scores = [score for label, score in bias_score[-10:][::-1]]

		labels = top_labels + ["..."] + bottom_labels
		values = top_scores + [0.001] + bottom_scores

		# Creating the bar chart
		fig, ax = plt.subplots(figsize=(15, 10))
		bars = ax.bar(top_labels, top_scores, color="blue", alpha=0.75)
		bars += ax.bar(["..."], [0.001], color="white")
		bars += ax.bar(bottom_labels, bottom_scores, color="red", alpha=0.75)
		# Making the "..." bar invisible
		bars[10].set_color("white")
		bars[10].set_edgecolor("white")

		# Adding text annotations on bars
		for bar, value in zip(bars, values):
			if value > 0.01:  # Avoid adding text on the "..." bar
				ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', fontsize=8, color='black' if bar.get_facecolor() == (0, 0, 0, 1) else 'black')

		# Formatting
		ax.axhline(y=0.5, color='green', linestyle='dashed')  # Reference line at 0.5
		ax.set_ylabel("BS(o)")
		ax.set_xticklabels(labels, rotation=20, ha="right")
		ax.set_title(f"Bias Score: {model}")
		#save
		plt.savefig(f"{model}_bias_score.png")

		final_cos_sim_nf = cos_sim_nf / len(results)
		final_cos_sim_nm = cos_sim_nm / len(results)
		all_results[model] = {
			"significant_objects": significant_objects,
			"final_cos_sim_nf": final_cos_sim_nf,
			"final_cos_sim_nm": final_cos_sim_nm
		}
		
	return all_results