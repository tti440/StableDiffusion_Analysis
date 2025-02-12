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
argparser.add_argument("--triples_json", type=str, default="example.json")
```
- **model_names**: The names of the Stable Diffusion models to analyze. The default value is three models: SD1.4, SD2.0, and SD2.1.
- **num_samples**: The number of samples to generate for each prompt. The default value is 1.
- **torch_dtype**: The data type to use for the PyTorch tensors. The default value is "torch.float16" for half-precision but can be changed to "torch.float32" for single-precision (or "torch.float64" for double-precision).
- **triple_json**: The json file should be a set of triples. This triple contain [neutral, feminine, masculine]. The example.json contains a set of 5 triples. 
