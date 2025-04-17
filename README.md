# **Stable Diffusion Analysis**  

This repository provides tools for analyzing Stable Diffusion-generated images using **object detection**, **segmentation**, and **similarity analysis**.

---

## **Installation Guide**  

### **Create a Conda Environment**  
create and activate the Conda environment:  

```bash
conda env create -f environment.yaml
conda activate sd_env
```
---

### **Install Dependencies**
install the dependencies:
```bash
pip install -e .
python -m spacy download en_core_web_sm
```

---

### **Manually Clone Required Repositories**  
The following dependencies **must be installed manually** 

#### **🔹 Install GroundingDINO**
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd /GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn
sed -i 's/value.type()/value.scalar_type()/g' ms_deform_attn_cuda.cu
sed -i 's/value.scalar_type().is_cuda()/value.is_cuda()/g' ms_deform_attn_cuda.cu
cd /StableDiffusion_Analysis
pip install -e GroundingDINO
```
##### **Download GroundingDINO Weights**
```bash
cd GroundingDINO
mkdir weights && cd weights
```
- **For Windows**  
  ```command prompt
  curl -L -o groundingdino_swint_ogc.pth "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
  ```
- **For Linux/macOS**  
  ```bash
  wget -O groundingdino_swint_ogc.pth "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
  ```

Then, return to the main directory:
```bash
cd ../..
```

Depending on your CUDA version and OS, you may encounter subprocess error while installing the dependencies. 
If you do, please refer the original repo for the instructions. 
**Reference:** [GroundingDINO Repository](https://github.com/IDEA-Research/GroundingDINO)

---

#### **🔹 Install SAM-2**
```bash
git clone https://github.com/facebookresearch/sam2.git sam2_repo
pip install -e sam2_repo
```
##### **Download SAM-2 Checkpoints**
```bash
cd sam2_repo/checkpoints && ./download_ckpts.sh
cd ../..
```
**Reference:** [SAM-2 Repository](https://github.com/facebookresearch/sam2)

---

#### **🔹 Install Recognize Anything**
```bash
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -e recognize-anything
```
**Reference:** [Recognize Anything Repository](https://github.com/xinyu1205/recognize-anything)

---

#### **🔹 Install DAAM**
```bash
git clone https://github.com/castorini/daam.git
pip install -e daam
```
**Reference:** [DAAM Repository](https://github.com/castorini/daam)

---

### **Usage**
Example usage of the `experiment.py` script is provided below.
Run the experiment with:
```bash
python stable_diffusion_analysis/experiment.py --num_samples=3
```
#### **Arguments:**
```python
argparser.add_argument("--model_names", type=str, default="SD1.4, SD2.0, SD2.1")
argparser.add_argument("--num_samples", type=int, default=1)
argparser.add_argument("--torch_dtype", type=str, default="torch.float16")
argparser.add_argument("--output_dir", type=str, default="output_data")
```
- **model_names**: The names of the Stable Diffusion models to analyze. The default value is three models: SD1.4, SD2.0, and SD2.1.
- **num_samples**: The number of samples to generate for each prompt. The default value is 1.
- **torch_dtype**: The data type to use for the PyTorch tensors. The default value is "torch.float16" for half-precision but can be changed to "torch.float32" for single-precision (or "torch.float64" for double-precision).
- **output_dir**: The directory where the plots will be saved. The default value is "output_data".

### **Prompt Data Choice**
Each prompt data should be stored in a json format and placed under **"datasets"** diretory in "stable_diffusion_analysis". The json file should contain a list of lists, where each inner list contains the prompts for neutral, feminine and masculine precisely in this order. Here is an example of coco.json:
```json
[
	[
		"a person at a table with a dog in a kitchen",
		"a woman at a table with a dog in a kitchen",
		"a man at a table with a dog in a kitchen"
	],
	[
		" a corner of a city street with people jumping and riding skateboards",
		" a corner of a city street with women jumping and riding skateboards",
		" a corner of a city street with men jumping and riding skateboards"
	]
]
```
The dataset should be specified in **"dataset.py"** in the form of a dictionary. The keys of the dictionary are the names of the datasets, and the values are the paths to the json files. Here is an example of how to specify the dataset in **"dataset.py"**:
```python
DATASETS = {
	"GCC": "gcc.json",
	"COCO": "coco.json",
	"TextCaps": "textcaps.json",
	"Flickr30k": "flickr30k.json",
	"Profession": "profession.json"
}
```
The name will be used to save the output files.

example.ipynb is provided to demonstrate results
