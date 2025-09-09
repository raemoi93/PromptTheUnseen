# Prompt the Unseen: Evaluating Visual-Language Alignment Beyond Supervision

## Acknowledgement

This repository is built upon [LLaVA](https://github.com/haotian-liu/LLaVA), 
and reuses parts of its codebase. We sincerely thank the authors for open-sourcing their work.

---

## Environment Setup

### Pull Docker image
```bash
docker pull pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
```

### Create and activate conda environment
```bash
conda create -n PromptTheUnseen python=3.10 -y
conda activate PromptTheUnseen
```

### Install base packages
```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

### Install additional libraries (for Llama3.2 training)
```bash
#install additional libraries for training. For Llama3.2
pip install deepspeed==0.16.5
pip install accelerate==0.26.1
pip install transformers==4.51.0
pip install peft==0.11.1
```

### Hugging Face access token
You have to export huggingface token to have access to weights of some models. You have to get access granted prior to run the experiments.  
Like below:
```bash
export HF_TOKEN=<Huggingface token that has access granted for the model weights.>
```

---

## Dataset Preparation

Download `train_seen.json` (for train) and `test_seen_mcqa.json` and `test_unseen_mcqa.json` (for evaluation) and place them under vg_datasets/

- train_seen.json (for train)  
  https://drive.google.com/file/d/15HJlAJGk8_BmZVoe-NrIcsAavFzkUwaT/view?usp=drive_link

- MCQA for test-seen (for evaluation)  
  https://drive.google.com/file/d/1b9z_Jg-Bla08yT_HkK1sRDHlxhCR4Ajg/view?usp=drive_link

- MCQA for test-unseen (for evaluation)  
  https://drive.google.com/file/d/1QFCcwjFEZDBudNo-o8u_F3MxKhiOYMIP/view?usp=drive_link

Download images from VG official website:

- official website: https://homes.cs.washington.edu/~ranjay/visualgenome/index.html  
- direct links for part1: https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip  
- direct link for part2: https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

---

## Checkpoints for Weights

The checkpoints are available at the following link:  
👉 [Google Drive Folder](https://drive.google.com/drive/folders/15mklHnurKdYMsffuif6LhYdkrulc1SLc?usp=sharing)

You will find three subfolders:

1. **vision_encoder_ablation_weights**  
   Models for the vision encoder ablation study, with the LM fixed to the default setting (*Llama-3.2-3B-Instruct*).  
   Contains weights for: **CLIP** (default), **Dinov2**, **ViT**, and **MAE**.

2. **lm_ablation_weights**  
   Models for the LM ablation study, with the vision encoder fixed to the default setting (**CLIP**).  
   Contains weights for: **Qwen3-1.7B**, **Qwen3-0.6B**, and **Llama-3.2-3B-Pretrain**.

3. **dataset_ablation_weights**  
   Models for the dataset ablation study. Includes weights trained with varying dataset ablation proportions using both **class-preserving** and **class-exclusive** methods.

---

### Notes
- All filenames are self-explanatory.  
- If any configuration is unclear from a filename, please open an issue and we will clarify.  
- To evaluate these models, use `scripts/test_default.sh` and refer to the **Evaluation** section below.


---

## Training

Run:
```bash
bash scripts/train_default.sh
```

You have to specify:
- `--image_folder` (the folder you downloaded the VisualGenome images)  
- `--output_dir` (the directory where the output weights will be saved)

Also you can modify:
- `--vision_tower` to `facebook/dinov2-large` (Dinov2), `facebook/webssl-mae300m-full2b-224` (MAE), `google/vit-large-patch16-224-in21k` (classification trained ViT) to train with various vision encoder backbones.
- `--model_name_or_path` to `meta-llama/Llama-3.2-3B` (Pretrain), `Qwen/Qwen3-1.7B` (Qwen3-1.7B), `Qwen/Qwen3-0.6B` (Qwen3-0.6B) to train with various LLM backbones

---

## Evaluation

Once you finish training, you can run evaluation on `test_seen_mcqa.json` and `test_unseen_mcqa.json` files.

Run:
```bash
bash scripts/test_default.sh
```

You have to specify:
- `--weight_path` (path to trained projection layer weights)
- `--output_path` (the path where prediction file will be saved)
- `--image_dir` (the folder you downloaded the VisualGenome images (identical path as during training))

Once the evaluation is done, you can run `compute_acc.py` to get macro averaged accuracy.

Exemplar prediction files are provided:
- https://drive.google.com/file/d/1ZYHM9ya-tbHRv1uiP0urxQ2B69xR1cbM/view?usp=drive_link  (default model on test_unseen_mcqa.json)
- https://drive.google.com/file/d/1tbwkuaCDa5zAewaIHX29MvOVBZqgKjbt/view?usp=drive_link  (default model on test_seen_mcqa.json)

Also you might have to modify 
- `--vision_tower` to `facebook/dinov2-large` (Dinov2), `facebook/webssl-mae300m-full2b-224` (MAE), `google/vit-large-patch16-224-in21k` (classification trained ViT) 
- `--model_name_or_path` to `meta-llama/Llama-3.2-3B` (Pretrain), `Qwen/Qwen3-1.7B` (Qwen3-1.7B), `Qwen/Qwen3-0.6B` (Qwen3-0.6B) 
according to your configuration

---

## FFN Analysis

### Download Prompts
- [Prompts for test-unseen](https://drive.google.com/file/d/1R37kA1lXaEpngYYdeb-XgNsCg0ToJV8R/view?usp=drive_link)  
- [Prompts for test-seen](https://drive.google.com/file/d/1273tSI_3r7FS1r8wVhbII-A2QIh7Vazt/view?usp=drive_link)  

### Run
```bash
bash scripts/ffn_analysis.sh
```

Before running, update the following arguments in the script:
- `--test_path`: path to the downloaded prompts  
- `--image_dir`: directory containing evaluation images  
- `--output_path`: directory where results will be saved  
- `--weight_path`: path to the model weights  
Optionally you may also modify `--vision_tower` and `--model_name_or_path` to replace vision encoder and LM backbone.

### Precomputed Results
- [FFN analysis results for test-seen](https://drive.google.com/file/d/1IX0hER3VmhafEOOKsAAz4a8fYESkMkz-/view?usp=drive_link)  
- [FFN analysis results for test-unseen](https://drive.google.com/file/d/1nE9tpzTkaxTG6-xeCLUuiPVwcaKK0fWv/view?usp=drive_link)  

After downloading and unzipping, you will find tens of thousands of JSON files. Each file corresponds to one forward pass of a prompt.

### JSON File Structure
- **`gt_synset`**: ground-truth class label  
- **`prompt`**: the actual prompt given to the model  
- **`top_toks_per_layer`**: top 9 extracted tokens from each layer  
  - Notably, after layer 15, these tokens begin to closely align with the ground-truth class label.  
- **`random_toks_per_layer`**: 9 tokens extracted from randomly selected values  
