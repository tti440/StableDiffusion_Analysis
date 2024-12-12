import torch
from models import get_model
from torch.nn.functional import cosine_similarity

def get_text_embedding(text: str, model: torch.nn.Module):
	'''
		Args:
			text: str
			model: torch.nn.Module
		Returns:
			torch.Tensor
		This function returns the text embedding of the given text in the CLIPTextEncoder in the model.
	'''
	prompt_ids = model.tokenizer(
		text,
		return_tensors="pt",
		padding="max_length",
		truncation=True,
		max_length=model.tokenizer.model_max_length,
	).input_ids.to(model.text_encoder.device)
	return model.text_encoder(prompt_ids)[0]

def get_image_embedding(image: torch.Tensor, model: torch.nn.Module):

def calculate_cosine_similarity(a: torch.Tensor, b: torch.Tensor):
	'''
		Args:
			a: torch.Tensor
			b: torch.Tensor
		Returns:
			float
		This function returns the cosine similarity between the two given vectors.
	'''
	return cosine_similarity(a, b).item()