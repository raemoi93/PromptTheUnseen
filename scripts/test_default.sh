
# you can change test_path to test_seen_mcqa.json
python demo_inference_vg_mcqa_plain.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type linear \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --weight_path <put the path to the weight file> \
  --test_path ./vg_datasets/test_seen_mcqa.json \
  --output_path <put the path to the output file> \
  --image_dir <put the path to the image directory> \
  --sample_prop 1.0 \
  --version plain \
  --model_max_length 2048 \
  --device cuda:6 \
  --bf16 
  