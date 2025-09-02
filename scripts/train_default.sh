# Download train_seen.json and place it under ./vg_datasets/
# To test with various Vision encoders, you can replace vision_tower to facebook/dinov2-large (Dinov2), facebook/webssl-mae300m-full2b-224 (MAE), google/vit-large-patch16-224-in21k (classification trained ViT)
# To test with various LLM backbones, you can replace model_name_or_path to meta-llama/Llama-3.2-3B (Pretrain), Qwen/Qwen3-1.7B (Qwen3-1.7B), Qwen/Qwen3-0.6B (Qwen3-0.6B)
# you can modify --include=localhost:4,5,6,7 according to your gpu setup

deepspeed --include=localhost:4,5,6,7 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --version llama_3_2_vg_object_centric \
    --data_path ./vg_datasets/train_seen.json \
    --image_folder <put the path to the image folder> \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type linear \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir <put the path to the output directory> \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb