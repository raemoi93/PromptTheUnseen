# This branch hosts the official code release for the paper “Prompt the Unseen: Evaluating Visual-Language Alignment Beyond Supervision.”

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

Download `train_seen.json` (for train) and `test_seen_mcqa.json` and `test_unseen_mcqa.json` (for evaluation).

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

## Checkpoints for weights

<!-- (No content was provided in the original text; header kept intentionally.) -->

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

---

## FFN analysis

Download prompts for test-seen and test-unseen:

- https://drive.google.com/file/d/1R37kA1lXaEpngYYdeb-XgNsCg0ToJV8R/view?usp=drive_link  (prompts from test-unseen)
- https://drive.google.com/file/d/1273tSI_3r7FS1r8wVhbII-A2QIh7Vazt/view?usp=drive_link  (prompts from test-seen)

Run:
```bash
bash scripts/ffn_analysis_test.sh
```

You have to modify:
- `--test_path` to the path where prompts are downloaded
- also modify for `--image_dir`, `--output_path` and `--weight_path`
