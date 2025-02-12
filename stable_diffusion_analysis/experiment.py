import os
import argparse
import json
import pickle
from typing import List, Union
import torch
from detect_objects import all_detect_objects, get_figure_cooccurrence
from similarity import all_similarities
from dependencies import all_dependencies, bias_score_dependencies
import pandas as pd
import matplotlib.pyplot as plt

def similarity_matrix(smilarity:dict):
	iterables = [[], ["neutral-feminine", "neutral-masculine"]]
	for model_name in smilarity.keys():
		iterables[0].append(model_name)
	index = pd.MultiIndex.from_product(iterables, names=["model", "target"])
	nf = []
	nm = []
	for model_name, data in smilarity.items():
		for target, value in data.items():
			target = target.split("_")[-1]
			if target == "nf":
				nf.append(value)
			elif target == "nm":
				nm.append(value)

	df = pd.DataFrame([nf,nm], index=index, columns=["Prompt", "Denoising", "SSIM", "Diff. Pix", "ResNet", "CLIP", "DINO", "Split-Product"])
	df.to_csv("similarity_matrix.csv")

def cooccurrence_similarity(cooccurrence):
	iterables = [[], ["neutral-feminine", "neutral-masculine"]]
	all_significant_objects = {}
	for model_name in cooccurrence.keys():
		iterables[0].append(model_name)
		all_significant_objects[model_name] = []
	index = pd.MultiIndex.from_product(iterables, names=["model", "target"])
	cooccurrence_data = []
	for model_name, data in cooccurrence.items():
		for target, value in data.items():
			if target == "significant_objects":
				all_significant_objects[model_name].append(value)
				continue
			cooccurrence_data.append(value)
	## change categories(columns) based on dataset
	df = pd.DataFrame(cooccurrence_data, index=index, columns=["Example"])
 
	df.to_csv("cooccurrence_similarity.csv")
	with open("all_significant_objects.json", "w") as f:
		json.dump(all_significant_objects, f)

def top10_most_frequent_objects(dependencies):
	for model, dependency in dependencies.items():
		eg = dependency["explicit_guided"]
		ig = dependency["implicit_guided"]
		ei = dependency["explicit_independent"]
		ii = dependency["implicit_independent"]
		hi = dependency["hidden"]
		eg = sorted(eg.items(), key=lambda x: x[1], reverse=True)
		ig = sorted(ig.items(), key=lambda x: x[1], reverse=True)
		ei = sorted(ei.items(), key=lambda x: x[1], reverse=True)
		ii = sorted(ii.items(), key=lambda x: x[1], reverse=True)
		hi = sorted(hi.items(), key=lambda x: x[1], reverse=True)

		fig, ax = plt.subplots(figsize=(15,10))
		if len(ig) >10:
			values = [x[1] for x in ig[:10]]
			labels = [x[0] for x in ig[:10]]
		else:
			values = [x[1] for x in eg]
			labels = [x[0] for x in eg]
		bars = ax.bar(labels, values, label="explicit_guided", color = "pink")
		for bar, value in zip(bars, values):
			ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', fontsize=8, color='black' if bar.get_facecolor() == (0, 0, 0, 1) else 'black')
		ax.set_ylabel("Cg(o,P)")
		ax.set_xticklabels(labels, rotation=30, ha="right")
		ax.set_title("Explicitly Guided Dependencies")
		plt.savefig(f"{model}_explicit_guided.png")
  
		fig, ax = plt.subplots(figsize=(15,10))
		if len(ig) >10:
			values = [x[1] for x in ig[:10]]
			labels = [x[0] for x in ig[:10]]
		else:
			values = [x[1] for x in ig]
			labels = [x[0] for x in ig]
		bars = ax.bar(labels, values, label="implicit_guided", color = "orange")
		for bar, value in zip(bars, values):
			ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', fontsize=8, color='black' if bar.get_facecolor() == (0, 0, 0, 1) else 'black')
		ax.set_ylabel("Cg(o,P)")
		ax.set_xticklabels(labels, rotation=30, ha="right")
		ax.set_title("Implicitly Guided Dependencies")
		plt.savefig(f"{model}_implicit_guided.png")
  
		fig, ax = plt.subplots(figsize=(15,10))
		if len(ig) >10:
			values = [x[1] for x in ei[:10]]
			labels = [x[0] for x in ei[:10]]
		else:
			values = [x[1] for x in ei]
			labels = [x[0] for x in ei]
		bars = ax.bar(labels, values, label="explicit_independent", color = "yellow")
		for bar, value in zip(bars, values):
			ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', fontsize=8, color='black' if bar.get_facecolor() == (0, 0, 0, 1) else 'black')
		ax.set_ylabel("Cg(o,P)")
		ax.set_xticklabels(labels, rotation=30, ha="right")
		ax.set_title("Explicitly Independent Dependencies")
		plt.savefig(f"{model}_explicit_independent.png")
		
		fig, ax = plt.subplots(figsize=(15,10))
		if len(ig) > 10:
			values = [x[1] for x in ii[:10]]
			labels = [x[0] for x in ii[:10]]
		else:
			values = [x[1] for x in ii]
			labels = [x[0] for x in ii]
		bars = ax.bar(labels, values, label="implicit_independent", color = "green")
		for bar, value in zip(bars, values):
			ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', fontsize=8, color='black' if bar.get_facecolor() == (0, 0, 0, 1) else 'black')
		ax.set_ylabel("Cg(o,P)")
		ax.set_xticklabels(labels, rotation=30, ha="right")
		ax.set_title("Implicitly Independent Dependencies")
		plt.savefig(f"{model}_implicit_independent.png")
		
		fig, ax = plt.subplots(figsize=(15,10))
		if len(ig) >10:
			values = [x[1] for x in hi[:10]]
			labels = [x[0] for x in hi[:10]]
		else:
			values = [x[1] for x in hi]
			labels = [x[0] for x in hi]
		bars = ax.bar(labels, values, label="hidden", color = "blue")
		for bar, value in zip(bars, values):
			ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', fontsize=8, color='black' if bar.get_facecolor() == (0, 0, 0, 1) else 'black')
		ax.set_ylabel("Cg(o,P)")
		ax.set_xticklabels(labels, rotation=30, ha="right")
		ax.set_title("Hidden Dependencies")
		plt.savefig(f"{model}_hidden.png")

def bias_dependency_plot(dependency_bias_score):
	for model, data in dependency_bias_score.items():
		implicit_guided = data["implicit_guided"]
		implicit_independent = data["implicit_independent"]
		implicit_guided = sorted(implicit_guided.items(), key=lambda x: x[1], reverse=True)
		implicit_independent = sorted(implicit_independent.items(), key=lambda x: x[1], reverse=True)
	
		top_labels = [label for label, score in implicit_guided[:10]]
		top_scores = [score for label, score in implicit_guided[:10]]
		bottom_labels = [label for label, score in implicit_guided[-10:][::-1]]
		bottom_scores = [score for label, score in implicit_guided[-10:][::-1]]
		labels = top_labels + ["..."] + bottom_labels
		values = top_scores + [0.001] + bottom_scores
		fig, ax = plt.subplots(figsize=(15,10))
		bars = ax.bar(top_labels, top_scores, color="blue", alpha=0.75)
		bars += ax.bar(["..."], [0.001], color="white")
		bars += ax.bar(bottom_labels, bottom_scores, color="red", alpha=0.75)
		bars[10].set_color("white")
		bars[10].set_edgecolor("white")
		for bar, value in zip(bars, values):
			if value > 0.01:
				ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', fontsize=8, color='black' if bar.get_facecolor() == (0, 0, 0, 1) else 'black')
		ax.axhline(y=0.5, color='green', linestyle='dashed')
		ax.set_ylabel("BS(o)")
		ax.set_xticklabels(labels, rotation=30, ha="right")
		ax.set_title(f"Bias Score of Implicitly Guided Dependencies: {model}")
		plt.savefig(f"{model}_implicit_guided_bias_score.png")
		top_labels = [label for label, score in implicit_independent[:10]]
		top_scores = [score for label, score in implicit_independent[:10]]
		bottom_labels = [label for label, score in implicit_independent[-10:][::-1]]
		bottom_scores = [score for label, score in implicit_independent[-10:][::-1]]
		labels = top_labels + ["..."] + bottom_labels
		values = top_scores + [0.001] + bottom_scores
		fig, ax = plt.subplots(figsize=(15,10))
		bars = ax.bar(top_labels, top_scores, color="blue", alpha=0.75)
		bars += ax.bar(["..."], [0.001], color="white")
		bars += ax.bar(bottom_labels, bottom_scores, color="red", alpha=0.75)
		bars[10].set_color("white")
		bars[10].set_edgecolor("white")
		for bar, value in zip(bars, values):
			if value > 0.01:
				ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha='center', fontsize=8, color='black' if bar.get_facecolor() == (0, 0, 0, 1) else 'black')
		ax.axhline(y=0.5, color='green', linestyle='dashed')
		ax.set_ylabel("BS(o)")
		ax.set_xticklabels(labels, rotation=30, ha="right")
		ax.set_title(f"Bias Score of Implicitly Independent Dependencies: {model}")
		plt.savefig(f"{model}_implicit_independent_bias_score.png")

def main(model_names:Union[str | List[str]], num_samples:int, torch_dtype:str, triples_json:str):
	if not os.path.exists("images"):
		os.makedirs("images")
	if isinstance(model_names, str):
		model_names = [model_names]
	triples = json.load(open(triples_json))
	#similarity = all_similarities(triples, model_names, torch_dtype, num_samples)
	#detect_objects_results = all_detect_objects(triples, model_names)
	detect_objects_results = {
		"SD1.4": pickle.load(open("detect_object_SD1.4_results.pkl", "rb")),
		"SD2.0": pickle.load(open("detect_object_SD2.0_results.pkl", "rb")),
		"SD2.1": pickle.load(open("detect_object_SD2.1_results.pkl", "rb"))
	}
	dependencies, guided_objects = all_dependencies(model_names, num_samples)
	dependency_bias_score = bias_score_dependencies(dependencies)
	#similarity_matrix(similarity)
	figure_cooccurrence = get_figure_cooccurrence(detect_objects_results)
	cooccurrence_similarity(figure_cooccurrence)
	top10_most_frequent_objects(dependencies)
	bias_dependency_plot(dependency_bias_score)
	with open("all_guided_objects.json", "w") as f:
		json.dump(guided_objects, f)
	
# argument - model_names, num_samples, torch_dtype, triples.json(List[List[str]]) 
if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--model_names", type=str, default=["SD1.4", "SD2.0", "SD2.1"])
	argparser.add_argument("--num_samples", type=int, default=1)
	argparser.add_argument("--torch_dtype", type=str, default=torch.float16)
	argparser.add_argument("--triples_json", type=str, default="example.json")
	args = argparser.parse_args()
	main(args.model_names, args.num_samples, args.torch_dtype, args.triples_json)