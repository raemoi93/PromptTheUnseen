# Modify vision_tower, model_name_or_path according to your setup
# 
python demo_inference_vg_attribution.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type linear \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --weight_path <path to the projection layer weights> \
  --test_path <path to the prompt file> \
  --output_path <directory where the results will be saved> \
  --sample_prop 1.0 \
  --version plain \
  --model_max_length 2048 \
  --device cuda:6 \
  --image_dir <path to the image directory> \
  --bf16 