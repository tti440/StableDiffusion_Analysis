import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import List, Union
import torch
import numpy as np
import cv2
from PIL import Image
from einops import rearrange
from torch.nn.functional import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform
from diffusers import StableDiffusionPipeline
import gdown
from pathlib import Path

def get_text_embedding(text: str, model: StableDiffusionPipeline):
	prompt_ids = model.tokenizer(
		text,
		return_tensors="pt",
		padding="max_length",
		truncation=True,
		max_length=model.tokenizer.model_max_length,
	).input_ids.to(model.text_encoder.device)
	return model.text_encoder(prompt_ids)[0]

# we apply half precision to the model for faster inference
# remove half() if you want to use double precision
def get_image_denoising(prompt: str, model: StableDiffusionPipeline):
	with torch.no_grad():
		# Initialize latent variables borrowed from the original library
		batch_size = 1 if isinstance(prompt, str) else len(prompt)
		num_images_per_prompt = 1
		height =  model.unet.config.sample_size * model.vae_scale_factor
		width =  model.unet.config.sample_size * model.vae_scale_factor
		num_inference_steps = 50
		guidance_scale = 7.5
		do_classifier_free_guidance = guidance_scale > 1.0
		callback_steps = 1
		device = model.device
		generator = torch.Generator(device).manual_seed(1024)
		model.check_inputs(prompt, height, width, callback_steps)
		text_embeddings = model._encode_prompt(
				prompt, device, num_images_per_prompt, do_classifier_free_guidance, None
			).half()
		# Prepare latents
		num_channels_latents = model.unet.in_channels
		latents = model.prepare_latents(
			batch_size * num_images_per_prompt,
			num_channels_latents,
			height,
			width,
			text_embeddings.dtype,
			device,
			generator,
			None,
		).half()

		# Prepare extra step kwargs (for scheduler)
		extra_step_kwargs = model.prepare_extra_step_kwargs(generator, 0.0)  # eta=0.0

		# Denoising loop: Store latents (z_t) at multiple steps
		model.scheduler.set_timesteps(num_inference_steps, device=device)
		timesteps = model.scheduler.timesteps
		denoising_space_embeddings = {}  # Store z_t embeddings at different steps
	
		for i, t in enumerate(timesteps):
				torch.cuda.empty_cache()
				# Prepare latents for classifier-free guidance
				latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
				latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

				# Predict noise residual
				noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.half()

				# Perform classifier-free guidance (CFG)
				if do_classifier_free_guidance:
					noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
					noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

				# Compute the next latent state (z_t -> z_t-1)
				latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

				# Store intermediate denoising space embeddings at key steps
				if i % 10 == 0 or i == len(timesteps) - 1:  # Save every 10 steps and final step
					denoising_space_embeddings[t.item()] = latents.clone().detach()

		# `z0` (Final latent after denoising)
		z0 = latents.clone().detach()
		torch.cuda.empty_cache()
	return z0

def get_ssim_score(img1: Image.Image, img2: Image.Image):
	img1 = transform_pillow2cv2(img1)
	img2 = transform_pillow2cv2(img2)
	gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
	ssim_score, ssim_map = ssim(gray1, gray2, full=True)
	return ssim_score, ssim_map

def get_diffpix(img1: Image.Image, img2: Image.Image, ssim_map: np.ndarray):
	img1 = transform_pillow2cv2(img1)
	img2 = transform_pillow2cv2(img2)
	gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
	_, thresh1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
	contours1, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	_, thresh2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
	contours2, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Combine contour pixels from both images
	contour_pixels = set()

	# Add contour pixels from image1
	for contour in contours1:
		for point in contour:
			x, y = point[0]
			contour_pixels.add((x, y))

	# Add contour pixels from image2
	for contour in contours2:
		for point in contour:
			x, y = point[0]
			contour_pixels.add((x, y))

	# Extract SSIM scores at combined contour pixels
	contour_ssim_scores = []
	for (x, y) in contour_pixels:
		if 0 <= x < ssim_map.shape[1] and 0 <= y < ssim_map.shape[0]:
			contour_ssim_scores.append(ssim_map[y, x])
   
	# Compute the ratio of high SSIM scores
	high_ssim_threshold = 0.8
	high_ssim_count = sum(1 for score in contour_ssim_scores if score > high_ssim_threshold)
	total_contour_pixels = len(contour_ssim_scores)

	# Calculate Diff. Pix.
	diff_pix = high_ssim_count / total_contour_pixels if total_contour_pixels > 0 else 0
	return diff_pix

def transform_pillow2cv2(img: Image.Image):
	image = img.convert("RGB")
	image = np.array(image)[:, :, ::-1].copy()
	return image

def calculate_cosine_similarity(a: torch.Tensor, b: torch.Tensor):
	# Flatten the tensors
	a = a.flatten()
	b = b.flatten()
	# normalize the tensors
	a = a / a.norm(dim=-1, keepdim=True)
	b = b / b.norm(dim=-1, keepdim=True)
	return cosine_similarity(a.flatten(), b.flatten(),dim=0).item()

def get_resnet_embedding(img: Image.Image, resnet50):
	# Load pre-trained ResNet-50 model
	resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])  # Remove the last FC layer
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	# Load and preprocess image
	input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

	# Extract embeddings
	with torch.no_grad():
		embedding_resnet = resnet50(input_tensor).squeeze().numpy()
	return embedding_resnet

def get_resnet_similarity(img1: Image.Image, img2: Image.Image, resnet50):
	embedding1 = get_resnet_embedding(img1, resnet50)
	embedding2 = get_resnet_embedding(img2, resnet50)
	cosine_similarity = calculate_cosine_similarity(torch.tensor(embedding1), torch.tensor(embedding2))
	return cosine_similarity

def get_clip_embedding(img: Image.Image, clip_model, clip_processor):
	input_tensor = clip_processor(images=np.array(img.convert("RGB")), return_tensors="pt")["pixel_values"]
	with torch.no_grad():
		image_embedding = clip_model.get_image_features(input_tensor.to("cuda"))
	image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
	return image_embedding

def get_clip_similarity(img1: Image.Image, img2: Image.Image, clip_model, clip_processor):
	embedding1 = get_clip_embedding(img1, clip_model, clip_processor)
	embedding2 = get_clip_embedding(img2, clip_model, clip_processor)
	cosine_similarity = calculate_cosine_similarity(embedding1, embedding2)
	return cosine_similarity

def get_dino_embedding(img: Image.Image, dino):
	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	dino.to("cuda")
	dino.eval()
	input_tensor = transform(img).unsqueeze(0).to("cuda")
	with torch.no_grad():
		image_embedding = dino(input_tensor)
	image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
	return image_embedding

def get_dino_similarity(img1: Image.Image, img2: Image.Image, dino):
	embedding1 = get_dino_embedding(img1, dino)
	embedding2 = get_dino_embedding(img2, dino)
	cosine_similarity = calculate_cosine_similarity(embedding1, embedding2)
	return cosine_similarity

def split_product(img1: Image.Image, img2: Image.Image):
	patch1 = extract_patches(img1)
	patch2 = extract_patches(img2)
	patch1 = patch1/patch1.norm(dim=-1, keepdim=True)
	patch2 = patch2/patch2.norm(dim=-1, keepdim=True)
	similarity = cosine_similarity(patch1, patch2, dim=-1)
	return similarity.max().item()

def extract_patches(image: Image.Image, patch_size:int =16):
	"""
	Extracts non-overlapping patches from an image.
	Returns: Tensor of shape (num_patches, patch_size * patch_size * 3)
	"""
	transform = transforms.Compose([
	transforms.ToTensor()  # Convert PIL image to tensor
	])
	image_tensor = transform(image)  # Shape: [C, H, W]

	# Ensure height and width are multiples of patch_size
	H, W = image_tensor.shape[1], image_tensor.shape[2]
	H = H - (H % patch_size)
	W = W - (W % patch_size)
	image_tensor = image_tensor[:, :H, :W]

	# Rearrange to extract patches
	patches = rearrange(image_tensor, "c (h ph) (w pw) -> (h w) (ph pw c)", ph=patch_size, pw=patch_size)
	return patches

def load_sam2():
	from sam2.build_sam import build_sam2
	from sam2.sam2_image_predictor import SAM2ImagePredictor
 
	sam2_checkpoint = "sam2_repo/checkpoints/sam2.1_hiera_large.pt"
	model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"
	sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
	predictor = SAM2ImagePredictor(sam2_model)
	return predictor

def load_ram():
	image_size = 384
	transform_ram = get_transform(image_size)
	file_path = Path("ram_swin_large_14m.pth")
	if not file_path.exists():
		gdown.download("https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth" \
		, "ram_swin_large_14m.pth", quiet=True)
	model = ram(pretrained="ram_swin_large_14m.pth", image_size=image_size, vit="swin_l")
	return model, transform_ram

def load_dino():
	model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", \
				   "GroundingDINO/weights/groundingdino_swint_ogc.pth")
	return model

def object_detections(img: Union[Image.Image, List[Image.Image]], sam2, ram_transform, ram, dino):
	if isinstance(img, Image.Image):
		img = [img]
	results = {
		#"labels": [],
		"nouns": [],
		#"boxes": [],
		"masks": []
	}
	for image in img:
		res, phrases, xyxy, masks = rgs_pipeline(image, sam2, ram_transform, ram, dino)
		#results["labels"].append(res)
		results["nouns"].append(phrases)
		#results["boxes"].append(xyxy)
		results["masks"].append(masks)
	return results

def rgs_pipeline(img: Image.Image, sam2, ram_transform, ram, dino):
    with torch.no_grad():  # Disable gradient calculation for inference
        torch.cuda.empty_cache()  # Free up memory before execution
        
        ram.to('cuda')
        image = ram_transform(img).unsqueeze(0).to('cuda')  # Move input to CUDA
        res = inference(image, ram)  # Run inference on RAM
        
        # Define transformation
        transform = T.Compose(
		[
			T.RandomResize([800], max_size=1333),
			T.ToTensor(),
			T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)
        
        image_source = img.convert("RGB")
        image = np.asarray(image_source)  # Convert image to NumPy array
        
        image_transformed, _ = transform(image_source, None)  # Apply transformation
        
        TEXT_PROMPT = " ,".join(res[0].split(" | "))  # Generate text prompt
        BOX_THRESHOLD = 0.35
        TEXT_THRESHOLD = 0.25

        # Object detection with DINO (ensure input tensor is on CUDA)
        boxes, logits, phrases = predict(
            model=dino,
            image=image_transformed.to('cuda'),  # Ensure transformed image is on CUDA
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        
        # Convert boxes to xyxy format
        h, w, _ = image.shape
        boxes = boxes.to('cuda') * torch.tensor([w, h, w, h], device='cuda')
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        xyxy[xyxy < 0] = 0  # Ensure all values are non-negative
        
        # Segment objects with SAM2
        sam2.set_image(image)
        masks, _, _ = sam2.predict(
            point_coords=None,
            point_labels=None,
            box=xyxy,
            multimask_output=False,
        )
        
        torch.cuda.empty_cache()  # Free memory after processing
        
        return res[0], phrases, xyxy, masks
